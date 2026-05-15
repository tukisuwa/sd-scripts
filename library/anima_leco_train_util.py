from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class AnimaPromptEmbeds:
    prompt_embeds: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    t5_input_ids: Optional[torch.Tensor]
    t5_attention_mask: Optional[torch.Tensor]


def encode_prompt_anima(tokenize_strategy, text_encoding_strategy, text_encoder, prompt: str) -> AnimaPromptEmbeds:
    tokens = tokenize_strategy.tokenize(prompt)
    prompt_embeds, attention_mask, t5_input_ids, t5_attention_mask = text_encoding_strategy.encode_tokens(
        tokenize_strategy, [text_encoder], tokens
    )
    return AnimaPromptEmbeds(prompt_embeds, attention_mask, t5_input_ids, t5_attention_mask)


def precompute_llm_adapter_prompt_embeds(anima, prompt_embeds: AnimaPromptEmbeds, device, dtype: torch.dtype) -> AnimaPromptEmbeds:
    prompt = prompt_embeds.prompt_embeds.to(device=device, dtype=dtype)
    attn_mask = prompt_embeds.attention_mask.to(device=device) if prompt_embeds.attention_mask is not None else None
    t5_input_ids = prompt_embeds.t5_input_ids.to(device=device, dtype=torch.long) if prompt_embeds.t5_input_ids is not None else None
    t5_mask = prompt_embeds.t5_attention_mask.to(device=device) if prompt_embeds.t5_attention_mask is not None else None

    if t5_input_ids is not None and getattr(anima, "use_llm_adapter", False) and hasattr(anima, "llm_adapter"):
        crossattn_emb = anima.llm_adapter(
            source_hidden_states=prompt,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=attn_mask,
        )
        if t5_mask is not None:
            crossattn_emb[~t5_mask.bool()] = 0
        return AnimaPromptEmbeds(crossattn_emb.detach().cpu(), None, None, None)

    return AnimaPromptEmbeds(prompt.detach().cpu(), None, None, None)


def repeat_prompt_embeds(prompt_embeds: AnimaPromptEmbeds, batch_size: int, device, dtype: torch.dtype) -> AnimaPromptEmbeds:
    def repeat(t: torch.Tensor, target_dtype: Optional[torch.dtype] = None):
        if t is None:
            return None
        if target_dtype is None:
            target_dtype = dtype if t.dtype.is_floating_point else t.dtype
        t = t.to(device=device, dtype=target_dtype)
        if t.shape[0] == batch_size:
            return t
        if t.shape[0] != 1:
            raise ValueError(f"unexpected prompt batch size: {t.shape[0]} != 1 or {batch_size}")
        return t.repeat(batch_size, *([1] * (t.ndim - 1)))

    return AnimaPromptEmbeds(
        repeat(prompt_embeds.prompt_embeds, dtype),
        repeat(prompt_embeds.attention_mask),
        repeat(prompt_embeds.t5_input_ids, torch.long),
        repeat(prompt_embeds.t5_attention_mask),
    )


def get_initial_latents_anima(batch_size: int, height: int, width: int, n_prompts: int = 1) -> torch.Tensor:
    noise = torch.randn((batch_size, 16, height // 8, width // 8), device="cpu")
    return noise.repeat(n_prompts, 1, 1, 1)


def apply_noise_offset_anima(latents: torch.Tensor, noise_offset: Optional[float]) -> torch.Tensor:
    if noise_offset is None:
        return latents
    noise = torch.randn((latents.shape[0], latents.shape[1], 1, 1), dtype=torch.float32, device="cpu")
    noise = noise.to(dtype=latents.dtype, device=latents.device)
    return latents + noise_offset * noise


def get_flow_sigmas(num_steps: int, flow_shift: float, device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    flow_shift = float(flow_shift)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)
    return sigmas


def _run_with_checkpoint(function, *args):
    if torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def predict_velocity_anima(
    anima,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: AnimaPromptEmbeds,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    batch_size = latents.shape[0]
    height = latents.shape[-2]
    width = latents.shape[-1]
    padding_mask = torch.zeros(batch_size, 1, height, width, dtype=latents.dtype, device=latents.device)
    model_input = latents.unsqueeze(2)
    timesteps = timestep.expand(batch_size).to(device=latents.device, dtype=latents.dtype)

    def run_model(x, t, embeds, attn_mask, t5_ids, t5_mask, padding):
        return anima(
            x,
            t,
            embeds,
            padding_mask=padding,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=attn_mask,
        )

    model_pred = _run_with_checkpoint(
        run_model,
        model_input,
        timesteps,
        prompt_embeds.prompt_embeds,
        prompt_embeds.attention_mask,
        prompt_embeds.t5_input_ids,
        prompt_embeds.t5_attention_mask,
        padding_mask,
    ).squeeze(2)

    if guidance_scale == 1.0:
        return model_pred

    raise ValueError("predict_velocity_anima expects already-combined CFG embeddings for guidance_scale != 1.0")


def predict_velocity_cfg_anima(
    anima,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    unconditional: AnimaPromptEmbeds,
    conditional: AnimaPromptEmbeds,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return predict_velocity_anima(anima, latents, timestep, conditional, guidance_scale=1.0)

    uncond_pred = predict_velocity_anima(anima, latents, timestep, unconditional, guidance_scale=1.0)
    cond_pred = predict_velocity_anima(anima, latents, timestep, conditional, guidance_scale=1.0)
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def diffusion_anima(
    anima,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    unconditional: AnimaPromptEmbeds,
    conditional: AnimaPromptEmbeds,
    total_timesteps: int,
    guidance_scale: float = 3.0,
) -> torch.Tensor:
    for step_index in range(total_timesteps):
        sigma = sigmas[step_index]
        next_sigma = sigmas[step_index + 1]
        model_pred = predict_velocity_cfg_anima(
            anima,
            latents,
            sigma,
            unconditional,
            conditional,
            guidance_scale=guidance_scale,
        )
        latents = latents + model_pred * (next_sigma - sigma)
        latents = latents.to(dtype=sigmas.dtype)
    return latents


def get_random_resolution_in_bucket_anima(bucket_resolution: int = 512) -> Tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2
    step = 16
    min_step = max(1, min_resolution // step)
    max_step = max(min_step, max_resolution // step)
    height = torch.randint(min_step, max_step + 1, (1,)).item() * step
    width = torch.randint(min_step, max_step + 1, (1,)).item() * step
    return height, width


def get_anima_resolution(prompt_setting) -> Tuple[int, int]:
    height, width = prompt_setting.get_resolution()
    if prompt_setting.dynamic_resolution and height == width:
        height, width = get_random_resolution_in_bucket_anima(height)
    height = max(16, height - height % 16)
    width = max(16, width - width % 16)
    return height, width
