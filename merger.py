import argparse
import json
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import train_util
from library.utils import setup_logging # このライブラリは別途必要です
import logging
import re
import math

setup_logging()
logger = logging.getLogger(__name__)

CLAMP_QUANTILE = 0.99
_LLOYD_MAX_CACHE = {}
_ROTATION_CACHE = {}

def load_metadata_from_safetensors(filename):
    with open(filename, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = f.read(header_len)
    header_json = json.loads(header)
    return header_json.get("__metadata__", {})

def precalculate_safetensors_hashes(tensors, metadata):
    if metadata is None:
        metadata = {}
    return train_util.precalculate_safetensors_hashes(tensors, metadata)

def get_base_module_name(key):
    # .lora_down.weight, .lora_up.weight, .alpha の前の部分をベース名とする
    for suffix in [".lora_down.weight", ".lora_up.weight", ".alpha"]:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None

def load_state_dict_and_collect_modules(file_name, dtype):
    metadata = {}
    base_modules = set()
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype)
        
        base_name = get_base_module_name(key)
        # lora_down.weight を代表としてモジュールを収集
        if base_name and key.endswith(".lora_down.weight"):
            base_modules.add(base_name)
            
    return sd, metadata, base_modules

def save_to_file(file_name, state_dict, metadata):
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata)
    else:
        torch.save(state_dict, file_name)

def _quantize_tensor(tensor, bits):
    original_dtype = tensor.dtype
    tensor_float = tensor.float()
    min_val = tensor_float.min()
    max_val = tensor_float.max()

    if max_val - min_val == 0:
        return torch.zeros_like(tensor, dtype=original_dtype)

    scale = (max_val - min_val) / (2**bits - 1)
    if scale == 0: # Avoid division by zero if max_val == min_val implies scale is 0
        return torch.zeros_like(tensor, dtype=original_dtype)
        
    zero_point = torch.round(-min_val / scale)
    quantized_tensor = torch.clamp(torch.round(tensor_float / scale + zero_point), 0, 2**bits - 1)
    dequantized_tensor = (quantized_tensor - zero_point) * scale

    return dequantized_tensor.to(original_dtype)

def _get_gaussian_lloyd_max_codebook(bits, device):
    cache_key = int(bits)
    cached = _LLOYD_MAX_CACHE.get(cache_key)
    if cached is None:
        levels = 2 ** int(bits)
        if levels < 2:
            raise ValueError("Turbo quantization requires at least 1 bit.")

        grid_size = 32768
        trunc = 6.0
        xs = torch.linspace(-trunc, trunc, steps=grid_size, dtype=torch.float64)
        pdf = torch.exp(-0.5 * xs.square()) / math.sqrt(2.0 * math.pi)
        q = torch.linspace(0.5 / levels, 1.0 - 0.5 / levels, steps=levels, dtype=torch.float64)
        centroids = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)

        for _ in range(64):
            boundaries = (centroids[:-1] + centroids[1:]) * 0.5
            new_centroids = centroids.clone()
            prev_boundary = -float("inf")
            for i in range(levels):
                next_boundary = boundaries[i] if i < levels - 1 else float("inf")
                mask = (xs >= prev_boundary) & (xs < next_boundary)
                weight = pdf[mask].sum()
                if weight > 0:
                    new_centroids[i] = (xs[mask] * pdf[mask]).sum() / weight
                prev_boundary = next_boundary
            if torch.max(torch.abs(new_centroids - centroids)) < 1e-7:
                centroids = new_centroids
                break
            centroids = new_centroids

        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        cached = (centroids.to(torch.float32).cpu(), boundaries.to(torch.float32).cpu())
        _LLOYD_MAX_CACHE[cache_key] = cached

    centroids, boundaries = cached
    return centroids.to(device=device), boundaries.to(device=device)

def _make_orthogonal_matrix(dim, generator, device):
    if dim <= 1:
        return torch.eye(dim, device=device, dtype=torch.float32)

    mat = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(mat, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(device=device)

def _get_rotation_matrix(dim, device, seed, style):
    cache_key = (int(dim), str(device), int(seed), style)
    cached = _ROTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(dim) * 104729)

    if style == "dense":
        rotation = _make_orthogonal_matrix(dim, generator, device)
    elif style == "block3":
        rotation = torch.eye(dim, device=device, dtype=torch.float32)
        start = 0
        while start < dim:
            block_dim = min(3, dim - start)
            rotation[start:start + block_dim, start:start + block_dim] = _make_orthogonal_matrix(block_dim, generator, device)
            start += block_dim
    else:
        raise ValueError(f"Unsupported rotation style: {style}")

    _ROTATION_CACHE[cache_key] = rotation
    return rotation

def _quantize_tensor_turbo_like(tensor, bits, seed, rotation_style):
    original_dtype = tensor.dtype
    tensor_float = tensor.float()
    if tensor_float.ndim == 1:
        tensor_float = tensor_float.unsqueeze(0)
        squeeze_back = True
    elif tensor_float.ndim == 2:
        squeeze_back = False
    else:
        raise ValueError("Turbo-style quantization currently expects a 1D or 2D tensor.")

    rows, dim = tensor_float.shape
    if dim == 0:
        return tensor.clone()
    if dim == 1:
        result = _quantize_tensor(tensor_float, bits).to(original_dtype)
        return result.squeeze(0) if squeeze_back else result

    norms = torch.linalg.norm(tensor_float, dim=1, keepdim=True)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    unit = tensor_float / safe_norms

    rotation = _get_rotation_matrix(dim, tensor_float.device, seed, rotation_style)
    rotated = unit @ rotation

    scaled = rotated * math.sqrt(dim)
    centroids, boundaries = _get_gaussian_lloyd_max_codebook(bits, tensor_float.device)
    bucket_ids = torch.bucketize(scaled.reshape(-1), boundaries)
    quantized_scaled = centroids[bucket_ids].reshape(rows, dim)

    dequantized_rotated = quantized_scaled / math.sqrt(dim)
    reconstructed = (dequantized_rotated @ rotation.transpose(0, 1)) * safe_norms
    reconstructed = torch.where(norms > 0, reconstructed, torch.zeros_like(reconstructed))
    reconstructed = reconstructed.to(original_dtype)

    if squeeze_back:
        reconstructed = reconstructed.squeeze(0)
    return reconstructed

def _quantize_tensor_with_method(tensor, bits, method, seed):
    if method == "uniform":
        return _quantize_tensor(tensor, bits)
    if method == "turbo":
        return _quantize_tensor_turbo_like(tensor, bits, seed=seed, rotation_style="dense")
    if method == "rotor":
        return _quantize_tensor_turbo_like(tensor, bits, seed=seed, rotation_style="block3")
    raise ValueError(f"Unsupported quantization method: {method}")

def _svd_lowrank_compat(mat, q, niter):
    if hasattr(torch, "svd_lowrank"):
        return torch.svd_lowrank(mat, q=int(q), niter=int(niter))
    if hasattr(torch.linalg, "svd_lowrank"):
        return torch.linalg.svd_lowrank(mat, q=int(q), niter=int(niter))
    raise AttributeError("No svd_lowrank implementation found in this torch build.")

def parse_module_filter_rules(rules_str_list):
    if not rules_str_list:
        return []
    
    parsed_rules = []
    for rule_str in rules_str_list:
        parts = rule_str.split(':')
        if not (2 <= len(parts) <= 3):
            raise ValueError(f"Invalid module filter format: {rule_str}. Expected 'PATTERN:ACTION[:STRENGTH]'.")
        
        pattern = parts[0]
        action = parts[1].lower()
        if action not in ["include", "exclude"]:
            raise ValueError(f"Invalid action '{action}' in filter rule '{rule_str}'. Must be 'include' or 'exclude'.")
        
        strength = 1.0
        if len(parts) == 3:
            if action == "exclude":
                logger.warning(f"Strength value ignored for 'exclude' action in rule: {rule_str}")
            else:
                try:
                    strength = float(parts[2])
                except ValueError:
                    raise ValueError(f"Invalid strength value '{parts[2]}' in filter rule '{rule_str}'. Must be a float.")
        
        parsed_rules.append({"pattern": pattern, "action": action, "strength": strength})
    return parsed_rules

def determine_final_module_configs(all_module_names, filter_rules, first_lora_modules=None, filter_by_first_lora=False):
    final_configs = {}

    # Initialize all modules
    for name in all_module_names:
        final_configs[name] = {"process": True, "strength": 1.0} 

    # Step 1: Apply --filter-by-first-lora if enabled
    if filter_by_first_lora and first_lora_modules is not None:
        logger.info(f"Applying filter: Only modules from the first LoRA will be initially included.")
        for name in all_module_names:
            if name not in first_lora_modules:
                final_configs[name]["process"] = False
                # logger.debug(f"Module {name} initially excluded by --filter-by-first-lora.")

    # Step 2: Apply custom filter rules sequentially (later rules override earlier ones)
    for rule in filter_rules:
        pattern_str = rule["pattern"]
        action = rule["action"]
        strength = rule["strength"]
        try:
            regex = re.compile(pattern_str)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern_str}': {e}. Skipping this rule.")
            continue

        for name in all_module_names:
            if regex.match(name):
                final_configs[name]["process"] = (action == "include")
                if action == "include":
                    final_configs[name]["strength"] = strength
                # logger.debug(f"Module {name} matched by rule '{pattern_str}': action={action}, strength={strength if action == 'include' else 'N/A'}. Settings updated.")
    
    return final_configs

def merge_lora_models(model_paths, ratios, new_rank, new_conv_rank, device, merge_dtype, 
                      no_clamp, quantize, quantize_bits, quantize_method, quantize_seed,
                      filter_by_first_lora, module_filter_rules_parsed,
                      svd_mode, svd_oversample, svd_niter):
    logger.info(f"New rank: {new_rank}, New conv rank (if specified): {new_conv_rank if new_conv_rank is not None else new_rank}")
    effective_new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank

    effective_no_clamp = no_clamp

    all_lora_base_modules = set()
    first_lora_base_modules = None
    lora_sds_metadata_cache = [] # Cache (sd, metadata, base_modules)
    try:
        torch_device = torch.device(device if device else "cpu")
    except (TypeError, ValueError) as err:
        logger.warning(f"Invalid device '{device}' ({err}). Falling back to CPU.")
        torch_device = torch.device("cpu")
        device = "cpu"

    logger.info("Collecting module information from all LoRAs...")
    for i, model_path in enumerate(tqdm(model_paths, desc="Scanning LoRAs")):
        sd, metadata, base_modules = load_state_dict_and_collect_modules(model_path, merge_dtype)
        all_lora_base_modules.update(base_modules)
        if i == 0:
            first_lora_base_modules = base_modules
        lora_sds_metadata_cache.append((sd, metadata, base_modules))

    logger.info(f"Found {len(all_lora_base_modules)} unique LoRA base modules across all models.")
    if filter_by_first_lora and first_lora_base_modules:
        logger.info(f"First LoRA contains {len(first_lora_base_modules)} base modules.")

    final_module_configs = determine_final_module_configs(
        all_lora_base_modules, 
        module_filter_rules_parsed, 
        first_lora_base_modules, 
        filter_by_first_lora
    )

    processed_module_count = sum(1 for mc in final_module_configs.values() if mc["process"])
    if processed_module_count == 0:
        logger.error("No LoRA modules selected for merging after applying all filters. Aborting.")
        return {}, {}
    logger.info(f"{processed_module_count} LoRA modules will be processed after filtering.")
    
    # Log final decisions for modules (can be verbose, use debug)
    for name, config in final_module_configs.items():
        if config["process"]:
            logger.debug(f"Final decision for {name}: Process=True, Strength={config['strength']:.2f}")
        else:
            logger.debug(f"Final decision for {name}: Process=False (excluded by filters)")


    merged_from_metadata = []
    processed_network_args_set = set()
    for i, (model_path, ratio) in enumerate(zip(model_paths, ratios)):
        logger.info(f"Preparing model: {model_path} with ratio: {ratio}")
        merged_from_metadata.append(os.path.basename(model_path))
        _, lora_metadata, _ = lora_sds_metadata_cache[i]
        if "ss_network_args" in lora_metadata:
            processed_network_args_set.add(f"{os.path.basename(model_path)}: {lora_metadata['ss_network_args']}")

    logger.info("Extracting new LoRA parameters via SVD...")
    merged_lora_sd = {}
    modules_to_process = [m for m, cfg in final_module_configs.items() if cfg["process"]]

    with torch.no_grad():
        for base_lora_module_name in tqdm(modules_to_process, desc="Processing modules"):
            module_config = final_module_configs[base_lora_module_name]

            accumulated_effect = None
            for i, (model_path, ratio) in enumerate(zip(model_paths, ratios)):
                lora_sd, _, _ = lora_sds_metadata_cache[i]

                down_key = base_lora_module_name + ".lora_down.weight"
                up_key = base_lora_module_name + ".lora_up.weight"
                alpha_key = base_lora_module_name + ".alpha"

                if down_key not in lora_sd:
                    continue
                if up_key not in lora_sd:
                    logger.warning(f"Missing '{up_key}' for module '{base_lora_module_name}' in {model_path}. Skipping this contribution.")
                    continue

                down_weight = lora_sd[down_key].to(torch_device)
                up_weight = lora_sd[up_key].to(torch_device)

                current_lora_rank = up_weight.size(1) if up_weight.ndim == 4 else down_weight.size(0)

                alpha = lora_sd.get(alpha_key)
                if alpha is None:
                    alpha = torch.tensor(float(current_lora_rank), dtype=merge_dtype)
                elif not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(float(alpha), dtype=merge_dtype)
                alpha = alpha.to(device=torch_device, dtype=merge_dtype)

                lora_scale_factor = alpha / current_lora_rank if current_lora_rank > 0 else alpha

                if len(down_weight.shape) != 4:
                    lora_effect = (up_weight @ down_weight) * lora_scale_factor
                elif down_weight.shape[2:4] == (1, 1):
                    lora_effect = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * lora_scale_factor
                else:
                    lora_effect = torch.einsum('orkh,rikh->oikh', up_weight, down_weight) * lora_scale_factor

                if accumulated_effect is None:
                    accumulated_effect = torch.zeros_like(lora_effect, dtype=merge_dtype, device=torch_device)

                accumulated_effect += ratio * lora_effect * module_config["strength"]

            if accumulated_effect is None:
                continue

            mat = accumulated_effect  # keep on merge device for SVD
            del accumulated_effect

            is_conv2d = len(mat.shape) == 4
            is_conv2d_ge_3x3 = False
            if is_conv2d:
                if mat.shape[2:4] != (1, 1):
                    is_conv2d_ge_3x3 = True
                out_dim, in_dim_orig, kh, kw = mat.shape
                mat_reshaped = mat.view(out_dim, -1)
            else:
                out_dim = mat.shape[0]
                in_dim_orig = mat.shape[1]
                mat_reshaped = mat

            current_module_rank = effective_new_conv_rank if is_conv2d_ge_3x3 else new_rank
            current_module_rank = min(current_module_rank, mat_reshaped.shape[0], mat_reshaped.shape[1])
            if current_module_rank == 0:
                logger.warning(f"Skipping SVD for {base_lora_module_name} due to effective rank 0.")
                continue

            mat_for_svd = mat_reshaped.to(torch.float32)
            if quantize:
                mat_for_svd = _quantize_tensor_with_method(
                    mat_for_svd,
                    bits=quantize_bits,
                    method=quantize_method,
                    seed=quantize_seed,
                ).to(torch.float32)

            V = None
            Vh = None
            if svd_mode == "speed":
                min_dim = min(mat_for_svd.shape[0], mat_for_svd.shape[1])
                q = current_module_rank + max(0, int(svd_oversample))
                q = max(current_module_rank, min(q, min_dim))
                try:
                    U, S, V = _svd_lowrank_compat(mat_for_svd, q=q, niter=int(svd_niter))
                    U = U[:, :current_module_rank]
                    S = S[:current_module_rank]
                    V = V[:, :current_module_rank]
                    lora_down_weight = V.transpose(0, 1)
                    lora_up_weight = U * S.unsqueeze(0)
                except Exception as e:
                    logger.warning(f"svd_lowrank failed for '{base_lora_module_name}' ({type(e).__name__}: {e}). Falling back to full SVD.")
                    U, S, Vh = torch.linalg.svd(mat_for_svd, full_matrices=False)
                    U = U[:, :current_module_rank]
                    S = S[:current_module_rank]
                    Vh = Vh[:current_module_rank, :]
                    lora_down_weight = Vh
                    lora_up_weight = U * S.unsqueeze(0)
            elif svd_mode == "resize_lora":
                out_size, in_size = mat_for_svd.shape
                if int(svd_niter) > 0 and out_size > 2048 and in_size > 2048:
                    q = min(2 * current_module_rank, out_size, in_size)
                    try:
                        U, S, V = _svd_lowrank_compat(mat_for_svd, q=q, niter=int(svd_niter))
                        Vh = V.transpose(0, 1)
                    except Exception as e:
                        logger.warning(f"svd_lowrank failed for '{base_lora_module_name}' ({type(e).__name__}: {e}). Falling back to full SVD.")
                        U, S, Vh = torch.linalg.svd(mat_for_svd, full_matrices=False)
                else:
                    U, S, Vh = torch.linalg.svd(mat_for_svd, full_matrices=False)
                U = U[:, :current_module_rank]
                S = S[:current_module_rank]
                Vh = Vh[:current_module_rank, :]
                lora_down_weight = Vh
                lora_up_weight = U * S.unsqueeze(0)
            else:
                U, S, Vh = torch.linalg.svd(mat_for_svd, full_matrices=False)
                U = U[:, :current_module_rank]
                S = S[:current_module_rank]
                Vh = Vh[:current_module_rank, :]
                lora_down_weight = Vh
                lora_up_weight = U * S.unsqueeze(0)

            if not effective_no_clamp:
                dist = torch.cat([lora_up_weight.flatten(), lora_down_weight.flatten()])
                hi_val = torch.quantile(dist.float(), CLAMP_QUANTILE)
                low_val = torch.quantile(dist.float(), 1.0 - CLAMP_QUANTILE)
                if not (hi_val == low_val and hi_val == 0):
                    lora_up_weight = lora_up_weight.clamp(low_val, hi_val)
                    lora_down_weight = lora_down_weight.clamp(low_val, hi_val)

            if is_conv2d:
                lora_down_weight = lora_down_weight.view(current_module_rank, in_dim_orig, kh, kw)
                lora_up_weight = lora_up_weight.view(out_dim, current_module_rank, 1, 1)

            merged_lora_sd[base_lora_module_name + ".lora_up.weight"] = lora_up_weight.to("cpu").contiguous().to(merge_dtype)
            merged_lora_sd[base_lora_module_name + ".lora_down.weight"] = lora_down_weight.to("cpu").contiguous().to(merge_dtype)
            merged_lora_sd[base_lora_module_name + ".alpha"] = torch.tensor(float(current_module_rank)).to("cpu").to(merge_dtype)

            del mat, mat_for_svd, mat_reshaped, U, S, V, Vh, lora_up_weight, lora_down_weight
            if torch_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_metadata = {}
    final_metadata["ss_network_module"] = "networks.lora"
    network_args = {"network_dim": new_rank, "network_alpha": float(new_rank)}
    if new_conv_rank is not None :
        network_args["conv_dim"] = new_conv_rank
        network_args["conv_alpha"] = float(new_conv_rank)
    final_metadata["ss_network_args"] = json.dumps(network_args)
    final_metadata["ss_merged_from"] = ", ".join(merged_from_metadata)
    if processed_network_args_set:
         final_metadata["ss_source_network_args"] = "; ".join(list(processed_network_args_set))
    
    final_metadata["ss_merge_filter_by_first_lora"] = str(filter_by_first_lora)
    final_metadata["ss_merge_module_filter_rules"] = json.dumps([
        f"{r['pattern']}:{r['action']}:{r['strength']}" for r in module_filter_rules_parsed
    ])

    logger.info(f"Final LoRA state dict has {len(merged_lora_sd)} keys.")
    return merged_lora_sd, final_metadata

def merge(args):
    if not args.models or not args.ratios:
        raise ValueError("Models and ratios must be provided.")
    if len(args.models) != len(args.ratios):
        raise ValueError("Number of models must be equal to number of ratios.")

    def str_to_dtype(p):
        if p is None: return None
        p_lower = p.lower()
        if p_lower == "float" or p_lower == "float32": return torch.float
        if p_lower == "fp16" or p_lower == "float16": return torch.float16
        if p_lower == "bf16" or p_lower == "bfloat16": return torch.bfloat16
        raise ValueError(f"Unsupported precision: {p}")

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision) if args.save_precision else merge_dtype

    try:
        module_filter_rules_parsed = parse_module_filter_rules(args.module_filter)
    except ValueError as e:
        logger.error(f"Error parsing module filter rules: {e}")
        return

    state_dict, metadata = merge_lora_models(
        args.models, args.ratios, args.new_rank,
        args.new_conv_rank, args.device, merge_dtype, args.no_clamp,
        args.quantize, args.quantize_bits, args.quantize_method, args.quantize_seed,
        args.filter_by_first_lora, module_filter_rules_parsed,
        args.svd_mode, args.svd_oversample, args.svd_niter
    )

    if not state_dict:
        logger.error("Merging resulted in an empty model state_dict. Exiting.")
        return

    for key in list(state_dict.keys()):
        value = state_dict[key]
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and value.dtype != save_dtype:
            state_dict[key] = value.to(save_dtype)

    effective_no_clamp = args.no_clamp
    merge_params_meta = {
        "models": [os.path.basename(p) for p in args.models], # Store only basenames for brevity
        "ratios": args.ratios,
        "new_rank": args.new_rank,
        "new_conv_rank": args.new_conv_rank,
        "quantize": args.quantize,
        "quantize_bits": args.quantize_bits,
        "quantize_method": args.quantize_method,
        "quantize_seed": args.quantize_seed,
        "no_clamp": args.no_clamp,
        "effective_no_clamp": effective_no_clamp,
        "svd_mode": args.svd_mode,
        "svd_oversample": args.svd_oversample,
        "svd_niter": args.svd_niter,
        "precision": args.precision,
        "save_precision": args.save_precision,
        "filter_by_first_lora": args.filter_by_first_lora,
        "module_filter": args.module_filter, # Store original filter strings
    }
    metadata["ss_merge_tool_version"] = "generic_lora_merger_0.2.3_quant_methods"
    metadata["ss_merge_params"] = json.dumps(merge_params_meta)

    logger.info(f"Calculating hashes and finalizing metadata...")
    model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash


    logger.info(f"Saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, metadata)
    logger.info("Done.")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Architecture-agnostic merger for sd-scripts style LoRA state dicts, with SVD recomposition and advanced filtering."
    )
    parser.add_argument(
        "--save_precision", type=str, default=None, choices=["float", "fp16", "bf16", "float32", "float16", "bfloat16"],
        help="Precision for saving the merged LoRA. Defaults to merging precision.",
    )
    parser.add_argument(
        "--precision", type=str, default="float", choices=["float", "fp16", "bf16", "float32", "float16", "bfloat16"],
        help="Precision for calculations during merging (float/float32 is generally recommended for stability).",
    )
    parser.add_argument(
        "--save_to", type=str, required=True,
        help="Path to save the new LoRA model (e.g., 'merged_lora.safetensors').",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Paths to the LoRA models to merge (e.g., 'lora1.safetensors' 'lora2.pt').",
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", required=True,
        help="Ratios for each model, in the same order as --models.",
    )
    parser.add_argument(
        "--new_rank", type=int, default=4,
        help="Target rank (dimension) for the output LoRA's linear layers.",
    )
    parser.add_argument(
        "--new_conv_rank", type=int, default=None,
        help="Target rank for convolutional LoRA layers (e.g., 3x3 convs). If None, uses --new_rank.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use for computations (e.g., 'cpu', 'cuda', 'mps'). Default: cpu.",
    )
    parser.add_argument(
        "--no_clamp", action="store_true",
        help="Disable clamping of SVD results to a quantile range.",
    )
    parser.add_argument(
        "--svd_mode", type=str, default="quality", choices=["quality", "speed", "resize_lora"],
        help="SVD computation mode. 'quality' uses full SVD. 'speed' uses randomized low-rank SVD (svd_lowrank). "
             "'resize_lora' follows sd-scripts resize_lora.py behavior: "
             "use svd_lowrank only for large matrices with q=min(2*rank, dims), otherwise full SVD.",
    )
    parser.add_argument(
        "--svd_oversample", type=int, default=4,
        help="Oversampling for randomized SVD when --svd_mode speed (q = rank + oversample). Default: 4.",
    )
    parser.add_argument(
        "--svd_niter", type=int, default=1,
        help="Power iterations for randomized SVD when --svd_mode speed or resize_lora. Higher improves accuracy but is slower. Default: 1.",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Quantize the merged delta weights before SVD.",
    )
    parser.add_argument(
        "--quantize_method", type=str, default="uniform", choices=["uniform", "turbo", "rotor"],
        help="Quantization method for --quantize. "
             "'uniform' is the original min/max uniform quantizer. "
             "'turbo' is a TurboQuant-inspired dense random rotation + Gaussian Lloyd-Max scalar quantizer. "
             "'rotor' is a RotorQuant-inspired block-3 rotation variant; this is an approximation, not a full Clifford algebra implementation.",
    )
    parser.add_argument(
        "--quantize_bits", type=int, default=8, choices=range(2,17), metavar="[2-16]",
        help="Number of bits for quantization if --quantize is enabled (default: 8).",
    )
    parser.add_argument(
        "--quantize_seed", type=int, default=42,
        help="Seed for randomized quantization methods such as 'turbo' and 'rotor'.",
    )
    parser.add_argument(
        "--quantize_note", action="store_true",
        help="Log a note describing the experimental nature of 'turbo' and 'rotor' quantizers.",
    )
    parser.add_argument(
        "--filter_by_first_lora", action="store_true",
        help="Filter modules to only include those present in the first LoRA model specified in --models. "
             "Other --module_filter rules are applied after this initial filtering if specified."
    )
    parser.add_argument(
        "--module_filter", type=str, nargs="*", default=[], # nargs="*" allows zero or more
        help="List of module filter rules. Each rule in 'PATTERN:ACTION[:STRENGTH]' format. "
             "PATTERN is a regex. ACTION is 'include' or 'exclude'. STRENGTH (float, default 1.0) applies to 'include'. "
             "Rules are applied sequentially, later rules can override earlier ones for the same module. "
             "Example: --module_filter \".*attn.*:include:0.8\" \".*text_encoder.*:exclude\""
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.quantize and not (2 <= args.quantize_bits <= 16) :
        parser.error("--quantize_bits must be between 2 and 16.")
    if args.svd_oversample < 0:
        parser.error("--svd_oversample must be >= 0.")
    if args.svd_niter < 0:
        parser.error("--svd_niter must be >= 0.")
    if args.quantize_note and args.quantize and args.quantize_method in ("turbo", "rotor"):
        logger.info(
            "Experimental quantizer enabled: '%s' is an approximation for pre-SVD tensor quantization and does not implement the full paper pipeline.",
            args.quantize_method,
        )

    merge(args)
