# Anima LoRA training script

import argparse
import json
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.ileco_text_encoder_conds = None
        self.ileco_prompt_pairs = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported. / fp8_baseとfp8_base_unetはサポートされていません。")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False  # Anima DiT does not support fp8_scaled

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        if args.ileco:
            assert (
                args.ileco_prompt_pairs is not None or args.ileco_target_prompt is not None
            ), "--ileco_target_prompt or --ileco_prompt_pairs is required when --ileco is enabled"
            assert args.ileco_loss_weight > 0, "--ileco_loss_weight must be greater than 0"
            assert args.reverse_weight > 0, "--reverse_weight must be greater than 0"
            if args.ileco_min_sigma is not None or args.ileco_max_sigma is not None:
                min_sigma = 0.0 if args.ileco_min_sigma is None else args.ileco_min_sigma
                max_sigma = 1.0 if args.ileco_max_sigma is None else args.ileco_max_sigma
                assert 0.0 <= min_sigma < max_sigma <= 1.0, "--ileco_min_sigma/max_sigma must satisfy 0 <= min < max <= 1"
            if not args.network_train_unet_only:
                logger.warning(
                    "--ileco trains through Anima DiT only. "
                    "Forcing --network_train_unet_only to avoid keeping the text encoder on GPU."
                )
                args.network_train_unet_only = True
            if not args.cache_latents:
                logger.warning(
                    "--ileco without --cache_latents keeps the VAE on GPU during training. "
                    "Enable --cache_latents or --cache_latents_to_disk to reduce VRAM usage."
                )
            if not args.gradient_checkpointing:
                logger.warning(
                    "--ileco without --gradient_checkpointing can use much more VRAM."
                )

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
        )
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
        )

        # Store unsloth preference so that when the base NetworkTrainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
            self.cache_ileco_text_encoder_outputs_if_needed(
                args, accelerator, text_encoders, text_encoding_strategy, tokenize_strategy, weight_dtype
            )

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    def load_ileco_prompt_pairs(self, args):
        if self.ileco_prompt_pairs is not None:
            return self.ileco_prompt_pairs

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
            normalized_pairs.append(
                {
                    "original": str(original or ""),
                    "target": str(target),
                    "weight": weight,
                    "multiplier": float(pair.get("multiplier", 1.0)),
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
                    }
                )
            normalized_pairs.extend(reverse_pairs)

        self.ileco_prompt_pairs = normalized_pairs
        return self.ileco_prompt_pairs

    def cache_ileco_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        text_encoders,
        text_encoding_strategy,
        tokenize_strategy,
        weight_dtype,
    ):
        if not args.ileco or self.ileco_text_encoder_conds is not None:
            return

        models = text_encoders if args.cache_text_encoder_outputs else self.get_models_for_text_encoding(args, accelerator, text_encoders)
        if models is None:
            raise ValueError("Text encoder models are not available for iLECO prompt encoding")

        pairs = self.load_ileco_prompt_pairs(args)
        prompts = []
        for pair in pairs:
            prompts.extend([pair["original"], pair["target"]])
        logger.info(f"cache iLECO Text Encoder outputs: {len(pairs)} prompt pair(s)")

        tokens_and_masks = tokenize_strategy.tokenize(prompts)
        tokens_and_masks = [t.to(accelerator.device) for t in tokens_and_masks]
        with torch.no_grad(), accelerator.autocast():
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        if args.full_fp16:
            encoded = [t.to(weight_dtype) if t is not None and t.dtype.is_floating_point else t for t in encoded]

        self.ileco_text_encoder_conds = []
        for i, pair in enumerate(pairs):
            original_index = i * 2
            target_index = original_index + 1
            self.ileco_text_encoder_conds.append(
                {
                    "original": [t[original_index : original_index + 1].detach() if t is not None else None for t in encoded],
                    "target": [t[target_index : target_index + 1].detach() if t is not None else None for t in encoded],
                    "weight": pair["weight"],
                    "multiplier": pair["multiplier"],
                }
            )

    def repeat_ileco_text_encoder_conds(self, conds, batch_size, device, weight_dtype):
        repeated = []
        for t in conds:
            if t is None:
                repeated.append(None)
                continue
            dtype = weight_dtype if t.dtype.is_floating_point else t.dtype
            t = t.to(device=device, dtype=dtype)
            if t.shape[0] == 1 and batch_size > 1:
                t = t.repeat(batch_size, *([1] * (t.ndim - 1)))
            elif t.shape[0] != batch_size:
                raise ValueError(f"Unexpected iLECO Text Encoder batch size: {t.shape[0]} != {batch_size}")
            repeated.append(t)
        return repeated

    def apply_limited_sigma_range(self, args, noise_scheduler, latents, noise, sigmas, min_sigma_arg, max_sigma_arg, dtype):
        if min_sigma_arg is None and max_sigma_arg is None:
            return None, None, None

        min_sigma = 0.0 if min_sigma_arg is None else min_sigma_arg
        max_sigma = 1.0 if max_sigma_arg is None else max_sigma_arg
        sigmas = min_sigma + sigmas * (max_sigma - min_sigma)
        timesteps = sigmas.flatten() * noise_scheduler.config.num_train_timesteps

        if args.ip_noise_gamma:
            xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
            if args.ip_noise_gamma_random_strength:
                ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
            else:
                ip_noise_gamma = args.ip_noise_gamma
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
        else:
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas

    def get_ileco_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        unet,
        network,
        weight_dtype,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        if self.ileco_text_encoder_conds is None:
            raise ValueError("iLECO Text Encoder outputs are not cached")
        if network is None or not hasattr(network, "set_multiplier"):
            raise ValueError("iLECO training requires a network with set_multiplier() support")

        if latents.ndim == 5:
            latents = latents.squeeze(2)
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        ranged_noisy_model_input, ranged_timesteps, ranged_sigmas = self.apply_limited_sigma_range(
            args,
            noise_scheduler,
            latents,
            noise,
            sigmas,
            args.ileco_min_sigma,
            args.ileco_max_sigma,
            weight_dtype,
        )
        if ranged_noisy_model_input is not None:
            noisy_model_input = ranged_noisy_model_input
            timesteps = ranged_timesteps
            sigmas = ranged_sigmas
        timesteps = timesteps / 1000.0

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)
        noisy_model_input = noisy_model_input.unsqueeze(2)

        pair_index = torch.randint(len(self.ileco_text_encoder_conds), (1,)).item()
        pair_conds = self.ileco_text_encoder_conds[pair_index]
        original_conds = self.repeat_ileco_text_encoder_conds(
            pair_conds["original"], bs, accelerator.device, weight_dtype
        )
        target_conds = self.repeat_ileco_text_encoder_conds(pair_conds["target"], bs, accelerator.device, weight_dtype)
        original_prompt_embeds, original_attn_mask, original_t5_input_ids, original_t5_attn_mask = original_conds[:4]
        target_prompt_embeds, target_attn_mask, target_t5_input_ids, target_t5_attn_mask = target_conds[:4]

        old_multiplier = getattr(network, "multiplier", 1.0)
        try:
            network.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                target = anima(
                    noisy_model_input,
                    timesteps,
                    target_prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=target_t5_input_ids,
                    target_attention_mask=target_t5_attn_mask,
                    source_attention_mask=target_attn_mask,
                )

            network.set_multiplier(pair_conds["multiplier"])
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    original_prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=original_t5_input_ids,
                    target_attention_mask=original_t5_attn_mask,
                    source_attention_mask=original_attn_mask,
                )
        finally:
            network.set_multiplier(old_multiplier)

        model_pred = model_pred.squeeze(2)
        target = target.detach().squeeze(2)

        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        weighting = weighting * args.ileco_loss_weight * pair_conds["weight"]

        return model_pred, target, timesteps, weighting

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        if args.ileco:
            return self.get_ileco_noise_pred_and_target(
                args,
                accelerator,
                noise_scheduler,
                latents,
                unet,
                network,
                weight_dtype,
                is_train=is_train,
            )

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
            :4
        ]  # ignore caption_dropout_rate which is not needed for training step

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        self.cache_ileco_text_encoder_outputs_if_needed(
            args, accelerator, text_encoders, text_encoding_strategy, tokenize_strategy, weight_dtype
        )

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        return super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
        )

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_ileco"] = args.ileco
        if args.ileco:
            metadata["ss_ileco_original_prompt"] = args.ileco_original_prompt
            metadata["ss_ileco_target_prompt"] = args.ileco_target_prompt
            metadata["ss_ileco_prompt_pairs"] = args.ileco_prompt_pairs
            metadata["ss_ileco_loss_weight"] = args.ileco_loss_weight
            metadata["ss_add_reverse_pairs"] = args.add_reverse_pairs
            metadata["ss_reverse_multiplier"] = args.reverse_multiplier
            metadata["ss_reverse_weight"] = args.reverse_weight
            metadata["ss_ileco_min_sigma"] = args.ileco_min_sigma
            metadata["ss_ileco_max_sigma"] = args.ileco_max_sigma

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        model = unet
        model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    # parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument("--ileco", action="store_true", help="enable dataset-backed iLECO prompt-to-prompt training")
    parser.add_argument("--ileco_original_prompt", type=str, default="", help="original prompt for single-pair iLECO")
    parser.add_argument("--ileco_target_prompt", type=str, default=None, help="target prompt for single-pair iLECO")
    parser.add_argument("--ileco_prompt_pairs", type=str, default=None, help="UTF-8 JSON file with iLECO prompt pairs")
    parser.add_argument("--ileco_loss_weight", type=float, default=1.0, help="loss multiplier for iLECO")
    parser.add_argument("--ileco_min_sigma", type=float, default=None, help="minimum sigma for iLECO timestep sampling")
    parser.add_argument("--ileco_max_sigma", type=float, default=None, help="maximum sigma for iLECO timestep sampling")
    parser.add_argument("--add_reverse_pairs", action="store_true", help="add reverse iLECO prompt pairs")
    parser.add_argument("--reverse_multiplier", type=float, default=-1.0, help="LoRA multiplier for reverse iLECO pairs")
    parser.add_argument("--reverse_weight", type=float, default=1.0, help="loss weight for reverse iLECO pairs")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
