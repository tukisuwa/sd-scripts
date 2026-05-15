import argparse
import importlib
import json
import os
import random
import sys

import torch
from accelerate.utils import set_seed
from tqdm import tqdm
import toml

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import anima_train_utils, anima_utils, custom_train_functions, strategy_anima, train_util
from library.anima_leco_train_util import (
    diffusion_anima,
    encode_prompt_anima,
    get_flow_sigmas,
    get_initial_latents_anima,
    precompute_llm_adapter_prompt_embeds,
    predict_velocity_cfg_anima,
    repeat_prompt_embeds,
)
from library.leco_train_util import PromptEmbedsCache, build_network_kwargs, normalize_resolution
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    train_util.add_training_arguments(parser, support_dreambooth=False)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser, support_weighted_captions=False)
    add_logging_arguments(parser)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "pt", "safetensors"],
        help="format to save the model (default is .safetensors)",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model")

    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_module", type=str, default="networks.lora_anima", help="network module to train")
    parser.add_argument("--network_dim", type=int, default=8, help="network rank")
    parser.add_argument("--network_alpha", type=float, default=1.0, help="network alpha")
    parser.add_argument("--network_dropout", type=float, default=None, help="network dropout")
    parser.add_argument("--network_args", type=str, default=None, nargs="*", help="additional network arguments")
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="unsupported for Anima iLECO; kept for compatibility",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="Anima iLECO always trains DiT LoRA only",
    )
    parser.add_argument("--training_comment", type=str, default=None, help="comment stored in metadata")
    parser.add_argument("--dim_from_weights", action="store_true", help="infer network dim from network_weights")
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for DiT LoRA")
    parser.add_argument(
        "--no_leco_cache_llm_adapter_outputs",
        action="store_true",
        help="do not precompute LLM Adapter outputs for prompt conditions",
    )
    parser.add_argument(
        "--ileco_latent_source",
        type=str,
        default="fixed_random",
        choices=["fixed_random", "dataset"],
        help="latent source for iLECO: fixed_random for prompt-only, dataset for dataset-backed training",
    )
    parser.add_argument("--ileco", action="store_true", help="deprecated no-op; anima_train_leco.py is iLECO-only")
    parser.add_argument("--ileco_original_prompt", type=str, default="", help="original prompt for single-pair iLECO")
    parser.add_argument("--ileco_target_prompt", type=str, default=None, help="target prompt for single-pair iLECO")
    parser.add_argument("--ileco_prompt_pairs", type=str, default=None, help="UTF-8 JSON file with iLECO prompt pairs")
    parser.add_argument("--ileco_loss_weight", type=float, default=1.0, help="loss multiplier for iLECO")
    parser.add_argument(
        "--ileco_guidance_scale",
        type=float,
        default=1.0,
        help="scale the iLECO target direction: original_base + scale * (target_base - original_base)",
    )
    parser.add_argument(
        "--ileco_denoising_steps",
        type=int,
        default=0,
        help="optional partial denoising steps before iLECO teacher/student prediction",
    )
    parser.add_argument(
        "--ileco_denoise_guidance_scale",
        type=float,
        default=1.0,
        help="guidance scale for iLECO partial denoising",
    )
    parser.add_argument("--ileco_resolution", type=int, default=512, help="default iLECO training resolution")
    parser.add_argument("--ileco_batch_size", type=int, default=1, help="default iLECO batch size")
    parser.add_argument("--ileco_min_sigma", type=float, default=0.0, help="minimum sigma for iLECO timestep sampling")
    parser.add_argument("--ileco_max_sigma", type=float, default=1.0, help="maximum sigma for iLECO timestep sampling")
    parser.add_argument("--add_reverse_pairs", action="store_true", help="add reverse iLECO prompt pairs")
    parser.add_argument("--reverse_multiplier", type=float, default=-1.0, help="LoRA multiplier for reverse iLECO pairs")
    parser.add_argument("--reverse_weight", type=float, default=1.0, help="loss weight for reverse iLECO pairs")

    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("--output_dir is required")
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required")
    if args.qwen3 is None:
        raise ValueError("--qwen3 is required")
    if not os.path.exists(args.pretrained_model_name_or_path):
        raise FileNotFoundError(f"--pretrained_model_name_or_path not found: {args.pretrained_model_name_or_path}")
    if not os.path.exists(args.qwen3):
        raise FileNotFoundError(f"--qwen3 not found: {args.qwen3}")
    if args.network_train_text_encoder_only:
        raise ValueError("Anima iLECO does not support text encoder LoRA training")
    if args.blocks_to_swap is not None and args.blocks_to_swap > 0:
        raise ValueError("Anima iLECO does not support --blocks_to_swap yet")
    if getattr(args, "fp8_base", False) or getattr(args, "fp8_base_unet", False):
        raise ValueError("Anima iLECO does not support fp8_base/fp8_base_unet")
    if args.save_model_as == "ckpt":
        raise ValueError("Anima LoRA weights can be saved as pt or safetensors")
    if args.ileco_prompt_pairs is None and args.ileco_target_prompt is None:
        raise ValueError("--ileco_target_prompt or --ileco_prompt_pairs is required")
    if args.ileco_loss_weight <= 0:
        raise ValueError("--ileco_loss_weight must be greater than 0")
    if args.ileco_guidance_scale <= 0:
        raise ValueError("--ileco_guidance_scale must be greater than 0")
    if args.ileco_denoising_steps < 0:
        raise ValueError("--ileco_denoising_steps must be 0 or greater")
    if args.reverse_weight <= 0:
        raise ValueError("--reverse_weight must be greater than 0")
    if not (0.0 <= args.ileco_min_sigma < args.ileco_max_sigma <= 1.0):
        raise ValueError("--ileco_min_sigma/max_sigma must satisfy 0 <= min < max <= 1")
    if args.cache_text_encoder_outputs or args.cache_text_encoder_outputs_to_disk:
        logger.warning("Anima iLECO encodes prompt pairs directly; text encoder output cache options are ignored")
    if args.max_train_epochs is not None:
        logger.warning("Anima iLECO is step-based; --max_train_epochs is ignored. Use --max_train_steps")
    if args.save_every_n_epochs is not None:
        logger.warning("Anima iLECO saves by steps only; --save_every_n_epochs is ignored. Use --save_every_n_steps")
    if args.llm_adapter_path is not None:
        logger.warning("--llm_adapter_path is currently ignored by Anima model loading; adapter weights must be in the DiT file")
    if args.v2 or args.v_parameterization or args.clip_skip is not None:
        logger.warning("Stable Diffusion specific options --v2/--v_parameterization/--clip_skip are ignored for Anima iLECO")
    if args.zero_terminal_snr or args.min_snr_gamma is not None:
        logger.warning("--zero_terminal_snr and --min_snr_gamma are not used for Anima Rectified Flow iLECO")
    if args.attn_mode == "sageattn":
        raise ValueError("sageattn is inference-only and cannot be used for Anima iLECO training")


def load_ileco_prompt_pairs(args: argparse.Namespace) -> list[dict]:
    if args.ileco_prompt_pairs is None:
        pairs = [
            {
                "original": args.ileco_original_prompt or "",
                "target": args.ileco_target_prompt,
                "weight": 1.0,
                "multiplier": 1.0,
            }
        ]
    else:
        with open(args.ileco_prompt_pairs, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", data) if isinstance(data, dict) else data

    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError("--ileco_prompt_pairs must contain at least one prompt pair")

    normalized_pairs = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"iLECO prompt pair #{i} must be an object")
        original = pair.get("original", pair.get("original_prompt", pair.get("source", pair.get("source_prompt", ""))))
        target = pair.get("target", pair.get("target_prompt", None))
        if target is None:
            raise ValueError(f"iLECO prompt pair #{i} does not have target or target_prompt")

        weight = float(pair.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"iLECO prompt pair #{i} weight must be greater than 0")

        resolution = normalize_resolution(pair.get("resolution", args.ileco_resolution))
        normalized_pairs.append(
            {
                "original": str(original or ""),
                "target": str(target),
                "weight": weight,
                "multiplier": float(pair.get("multiplier", 1.0)),
                "resolution": resolution,
                "batch_size": int(pair.get("batch_size", args.ileco_batch_size)),
            }
        )

    if args.add_reverse_pairs:
        reverse_pairs = []
        for pair in normalized_pairs:
            reverse_pairs.append(
                {
                    "original": pair["target"],
                    "target": pair["original"],
                    "weight": args.reverse_weight,
                    "multiplier": args.reverse_multiplier,
                    "resolution": pair["resolution"],
                    "batch_size": pair["batch_size"],
                }
            )
        normalized_pairs.extend(reverse_pairs)

    return normalized_pairs


def get_pair_resolution(pair: dict) -> tuple[int, int]:
    resolution = pair["resolution"]
    if isinstance(resolution, tuple):
        height, width = resolution
    else:
        height = width = int(resolution)
    height = max(16, height - height % 16)
    width = max(16, width - width % 16)
    return height, width


def sample_ileco_sigma(args: argparse.Namespace, device, dtype: torch.dtype) -> torch.Tensor:
    sigma = torch.rand((), device=device, dtype=dtype)
    return args.ileco_min_sigma + sigma * (args.ileco_max_sigma - args.ileco_min_sigma)


def save_ileco_weights(accelerator, network, args: argparse.Namespace, save_dtype, ileco_pairs, global_step: int, last: bool = False):
    os.makedirs(args.output_dir, exist_ok=True)
    ext = ".pt" if args.save_model_as == "pt" else ".safetensors"
    ckpt_name = train_util.get_last_ckpt_name(args, ext) if last else train_util.get_step_ckpt_name(args, ext, global_step)
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    metadata = None
    if not args.no_metadata:
        metadata = {
            "ss_network_module": args.network_module,
            "ss_network_dim": str(args.network_dim),
            "ss_network_alpha": str(args.network_alpha),
            "ss_base_model_version": "anima_preview",
            "ss_leco_model_type": "anima_ileco",
            "ss_ileco": "True",
            "ss_ileco_pair_count": str(len(ileco_pairs)),
            "ss_ileco_prompt_pairs": os.path.basename(args.ileco_prompt_pairs) if args.ileco_prompt_pairs else "",
            "ss_ileco_loss_weight": str(args.ileco_loss_weight),
            "ss_ileco_min_sigma": str(args.ileco_min_sigma),
            "ss_ileco_max_sigma": str(args.ileco_max_sigma),
            "ss_add_reverse_pairs": str(args.add_reverse_pairs),
            "ss_reverse_multiplier": str(args.reverse_multiplier),
            "ss_reverse_weight": str(args.reverse_weight),
        }
        if args.training_comment:
            metadata["ss_training_comment"] = args.training_comment
        metadata["ss_ileco_preview"] = json.dumps(ileco_pairs[:16], ensure_ascii=False)

    accelerator.unwrap_model(network).save_weights(ckpt_file, save_dtype, metadata)
    logger.info(f"saved model to: {ckpt_file}")


def get_cli_option_value(argv: list[str], option_name: str) -> str | None:
    prefix = option_name + "="
    for i, arg in enumerate(argv):
        if arg == option_name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def remove_cli_option(argv: list[str], option_name: str) -> list[str]:
    filtered = []
    skip_next = False
    prefix = option_name + "="
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == option_name:
            skip_next = True
            continue
        if arg.startswith(prefix):
            continue
        filtered.append(arg)
    return filtered


def config_requests_dataset_backed(config_file: str | None) -> bool:
    if not config_file:
        return False

    config_path = config_file + ".toml" if not config_file.endswith(".toml") else config_file
    if not os.path.exists(config_path):
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    flat_config = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            flat_config.update(value)
        else:
            flat_config[key] = value

    return flat_config.get("ileco_latent_source") == "dataset" or flat_config.get("dataset_config") is not None


def should_use_dataset_backed_ileco(argv: list[str]) -> bool:
    latent_source = get_cli_option_value(argv, "--ileco_latent_source")
    if latent_source == "dataset":
        return True
    if get_cli_option_value(argv, "--dataset_config") is not None:
        return True
    return config_requests_dataset_backed(get_cli_option_value(argv, "--config_file"))


def run_dataset_backed_ileco(argv: list[str]) -> None:
    from anima_train_network import AnimaNetworkTrainer, setup_parser as setup_dataset_parser

    dataset_argv = remove_cli_option(argv, "--ileco_latent_source")
    if "--ileco" not in dataset_argv:
        dataset_argv.append("--ileco")

    parser = setup_dataset_parser()
    args = parser.parse_args(dataset_argv)
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    args.ileco = True

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    trainer = AnimaNetworkTrainer()
    trainer.train(args)


def load_anima_for_leco(args: argparse.Namespace, weight_dtype: torch.dtype, accelerator):
    logger.info("Loading Qwen3 text encoder...")
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    text_encoder.eval()

    attn_mode = "torch"
    if args.xformers:
        attn_mode = "xformers"
    if args.attn_mode is not None:
        attn_mode = args.attn_mode
    if attn_mode == "sdpa":
        attn_mode = "torch"

    logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn={args.split_attn}...")
    anima = anima_utils.load_anima_model(
        accelerator.device,
        args.pretrained_model_name_or_path,
        attn_mode,
        args.split_attn,
        accelerator.device,
        weight_dtype,
        False,
    )
    anima.requires_grad_(False)
    anima.to(accelerator.device, dtype=weight_dtype)
    anima.train()

    return text_encoder, anima


def main():
    if should_use_dataset_backed_ileco(sys.argv[1:]):
        run_dataset_backed_ileco(sys.argv[1:])
        return

    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train_util.verify_training_args(args)
    validate_args(args)

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    set_seed(args.seed)

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    ileco_pairs = load_ileco_prompt_pairs(args)
    logger.info(f"loaded {len(ileco_pairs)} iLECO prompt pairs")

    text_encoder, anima = load_anima_for_leco(args, weight_dtype, accelerator)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    network_module = importlib.import_module(args.network_module)
    net_kwargs = build_network_kwargs(args)
    train_llm_adapter = str(net_kwargs.get("train_llm_adapter", "false")).lower() == "true"

    prompt_cache = PromptEmbedsCache()
    unique_prompts = sorted({prompt for pair in ileco_pairs for prompt in (pair["original"], pair["target"])})
    with torch.no_grad(), accelerator.autocast():
        for prompt in unique_prompts:
            encoded = encode_prompt_anima(tokenize_strategy, text_encoding_strategy, text_encoder, prompt)
            if not args.no_leco_cache_llm_adapter_outputs and not train_llm_adapter:
                encoded = precompute_llm_adapter_prompt_embeds(anima, encoded, accelerator.device, weight_dtype)
            prompt_cache[prompt] = encoded

    if not args.no_leco_cache_llm_adapter_outputs and not train_llm_adapter:
        logger.info("precomputed Anima LLM Adapter outputs for iLECO prompts")
    elif train_llm_adapter:
        logger.warning("train_llm_adapter=true disables LECO LLM Adapter output precomputation")

    text_encoder.to("cpu")
    clean_memory_on_device(accelerator.device)

    text_encoders = [text_encoder]
    if args.dim_from_weights:
        if args.network_weights is None:
            raise ValueError("--dim_from_weights requires --network_weights")
        network, _ = network_module.create_network_from_weights(1.0, args.network_weights, None, text_encoders, anima, **net_kwargs)
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            text_encoders,
            anima,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )

    network.apply_to(text_encoders, anima, apply_text_encoder=False, apply_unet=True)
    network.set_multiplier(0.0)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        logger.info(f"loaded network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        anima.enable_gradient_checkpointing(cpu_offload=getattr(args, "cpu_offload_checkpointing", False))
        network.enable_gradient_checkpointing()

    unet_lr = args.unet_lr if args.unet_lr is not None else args.learning_rate
    if hasattr(network, "prepare_optimizer_params"):
        trainable_params, _ = network.prepare_optimizer_params(None, unet_lr, args.learning_rate)
    else:
        trainable_params, _ = network.prepare_optimizer_params_with_multiple_te_lrs(None, unet_lr, args.learning_rate)
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)
    accelerator.unwrap_model(network).prepare_grad_etc(text_encoders, anima)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    optimizer_train_fn, _ = train_util.get_optimizer_train_eval_fn(optimizer, args)
    optimizer_train_fn()
    train_util.init_trackers(accelerator, args, "anima_ileco_train")

    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    fixed_ileco_latents = {}

    while global_step < args.max_train_steps:
        with accelerator.accumulate(network):
            optimizer.zero_grad(set_to_none=True)

            network_multiplier = accelerator.unwrap_model(network)
            pair = ileco_pairs[torch.randint(0, len(ileco_pairs), (1,)).item()]
            height, width = get_pair_resolution(pair)
            batch_size = pair["batch_size"]
            latent_key = (batch_size, height, width)
            if latent_key not in fixed_ileco_latents:
                fixed_ileco_latents[latent_key] = get_initial_latents_anima(batch_size, height, width, 1)
            model_input = fixed_ileco_latents[latent_key].to(accelerator.device, dtype=weight_dtype)

            original = repeat_prompt_embeds(prompt_cache[pair["original"]], batch_size, accelerator.device, weight_dtype)
            target_prompt = repeat_prompt_embeds(prompt_cache[pair["target"]], batch_size, accelerator.device, weight_dtype)

            if args.ileco_denoising_steps > 0:
                denoise_sigmas = get_flow_sigmas(
                    args.ileco_denoising_steps, args.discrete_flow_shift, accelerator.device, weight_dtype
                )
                network_multiplier.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    model_input = diffusion_anima(
                        anima,
                        model_input,
                        denoise_sigmas,
                        original,
                        original,
                        total_timesteps=args.ileco_denoising_steps,
                        guidance_scale=args.ileco_denoise_guidance_scale,
                    )

            current_sigma = sample_ileco_sigma(args, accelerator.device, weight_dtype)

            network_multiplier.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                original_base = predict_velocity_cfg_anima(
                    anima, model_input, current_sigma, original, original, guidance_scale=1.0
                )
                target_base = predict_velocity_cfg_anima(
                    anima, model_input, current_sigma, target_prompt, target_prompt, guidance_scale=1.0
                )
                target = original_base + args.ileco_guidance_scale * (target_base - original_base)

            network_multiplier.set_multiplier(pair["multiplier"])
            with accelerator.autocast():
                model_pred = predict_velocity_cfg_anima(
                    anima, model_input, current_sigma, original, original, guidance_scale=1.0
                )
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=(1, 2, 3)).mean() * args.ileco_loss_weight * pair["weight"]

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            network_multiplier = accelerator.unwrap_model(network)
            network_multiplier.set_multiplier(0.0)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "sigma": current_sigma.detach().float().item(),
            }
            logs["network_multiplier"] = pair["multiplier"]
            logs["pair_weight"] = pair["weight"]
            logs["ileco_guidance_scale"] = args.ileco_guidance_scale
            logs["ileco_fixed_latent"] = 1.0
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(loss=f"{logs['loss']:.4f}")

            if args.save_every_n_steps and global_step % args.save_every_n_steps == 0 and global_step < args.max_train_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_ileco_weights(accelerator, network, args, save_dtype, ileco_pairs, global_step, last=False)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_ileco_weights(accelerator, network, args, save_dtype, ileco_pairs, global_step, last=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
