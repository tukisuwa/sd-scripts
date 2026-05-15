# Anima iLECO Training / Anima iLECO 学習

This document describes the prompt-only iLECO training script for Anima, `anima_train_leco.py`.

<details>
<summary>日本語</summary>

このドキュメントは、Anima 向けの prompt-only iLECO 学習スクリプト `anima_train_leco.py` について説明します。

</details>

## Overview / 概要

`anima_train_leco.py` implements prompt-to-prompt iLECO for Anima.

The core training signal is:

```txt
teacher: LoRA OFF + target prompt
student: LoRA ON  + original prompt
loss:    student prediction -> teacher prediction
```

iLECO directly distills the base Anima prediction for the target prompt into the LoRA-applied original prompt.

<details>
<summary>日本語</summary>

`anima_train_leco.py` は、Anima 向けの prompt-to-prompt iLECO を実装しています。

学習信号の中核は以下です。

```txt
teacher: LoRA OFF + target prompt
student: LoRA ON  + original prompt
loss:    student prediction -> teacher prediction
```

iLECO は、target prompt に対する base Anima の prediction を、LoRA 適用済み original prompt 側へ直接蒸留します。

</details>

## Prompt Pairs / プロンプトペア

Use a UTF-8 JSON file. `--ileco` is accepted for compatibility, but the script is iLECO-only and does not need a mode switch. Use `--ileco_latent_source=dataset` with `--dataset_config` for dataset-backed iLECO.

```json
{
  "pairs": [
    {
      "original": "1girl, red hair",
      "target": "1girl, blue hair",
      "weight": 1.0,
      "multiplier": 1.0,
      "resolution": 512,
      "batch_size": 1
    },
    {
      "original": "1girl",
      "target": "1girl",
      "weight": 0.5,
      "multiplier": 1.0
    }
  ]
}
```

Example:

```powershell
accelerate launch --num_cpu_threads_per_process 1 anima_train_leco.py `
  --ileco_prompt_pairs="path\to\anima_ileco_pairs.json" `
  --pretrained_model_name_or_path="path\to\anima_model.safetensors" `
  --qwen3="path\to\qwen3.safetensors" `
  --output_dir="path\to\output_dir" `
  --output_name="anima_ileco_test" `
  --save_model_as=safetensors `
  --network_module=networks.lora_anima `
  --network_dim=8 `
  --network_alpha=8 `
  --learning_rate=1e-4 `
  --optimizer_type="AdamW8bit" `
  --lr_scheduler="constant" `
  --max_train_steps=500 `
  --mixed_precision="bf16"
```

Optional reverse pairs:

```txt
--add_reverse_pairs
--reverse_multiplier=-1.0
--reverse_weight=1.0
```

The following options are tuning knobs, not the default recommendation. Start without them, then add them only when the edit direction is too weak, too strong, or you want to restrict the trained timestep range:

```txt
--ileco_guidance_scale=3.0
--ileco_denoising_steps=4
--ileco_min_sigma=0.2
--ileco_max_sigma=0.8
```

`--ileco_guidance_scale` builds the target as:

```txt
target = original_base + scale * (target_base - original_base)
```

For stronger but riskier edits, try `--ileco_guidance_scale=5.0`. `--ileco_denoising_steps` adds extra teacher/student denoising work and can slow training, so keep the default unless the basic prompt-only target is not enough.

<details>
<summary>日本語</summary>

`anima_train_leco.py` は prompt-to-prompt の teacher/student 学習のみを行います。`--ileco` は互換性のため指定しても構いませんが、現在は mode switch ではありません。

```txt
teacher: LoRA OFF + target prompt
student: LoRA ON  + original prompt
loss:    student prediction -> teacher prediction
```

iLECO は、target prompt に対する base Anima の prediction を、LoRA 適用済み original prompt 側へ直接蒸留します。

prompt pair は UTF-8 JSON で指定します。dataset-backed iLECO を使う場合は、`--dataset_config` と一緒に `--ileco_latent_source=dataset` を指定してください。

逆方向ペアを自動追加する場合は以下を使います。

```txt
--add_reverse_pairs
--reverse_multiplier=-1.0
--reverse_weight=1.0
```

以下のオプションは標準推奨ではなく、必要に応じて使う調整用です。まずは付けずに開始し、edit 方向が弱い、強すぎる、または学習する timestep 範囲を絞りたい場合だけ追加してください。

```txt
--ileco_guidance_scale=3.0
--ileco_denoising_steps=4
--ileco_min_sigma=0.2
--ileco_max_sigma=0.8
```

`--ileco_guidance_scale` は以下の形で target を作ります。

```txt
target = original_base + scale * (target_base - original_base)
```

より強いが崩れやすい edit では `--ileco_guidance_scale=5.0` を試せます。`--ileco_denoising_steps` は teacher/student の追加 denoise を行うため学習が遅くなるので、通常の prompt-only target で足りない場合だけ使ってください。

</details>

## Latent Sources / latent source

There are two supported Anima iLECO variants:

| Implementation | Latent source | Behavior |
|---|---|---|
| `anima_train_leco.py` | internal prompt-only latent | prompt-only iLECO |
| `anima_train_leco.py --ileco_latent_source=dataset` | dataset image latents + noise | dataset-backed iLECO |

The dataset-backed path does not use the training image content as the semantic target. The image mainly provides an image-like latent, resolution/bucket behavior, and the base point for adding noise.

If prompt-only iLECO is too weak for a concept, use the dataset-backed iLECO path instead. The training image does not need to be a perfect target image, but it should provide a useful structure for the concept. For hair-color edits, portrait/person images are a better latent base than unrelated landscapes or abstract images.

Prompt-only iLECO can be tested with:

```powershell
accelerate launch --num_cpu_threads_per_process 1 anima_train_leco.py `
  --ileco_prompt_pairs="path\to\anima_ileco_pairs.json" `
  --pretrained_model_name_or_path="path\to\anima_model.safetensors" `
  --qwen3="path\to\qwen3.safetensors" `
  --output_dir="path\to\output_dir" `
  --output_name="anima_ileco_prompt_test" `
  --save_model_as=safetensors `
  --network_module=networks.lora_anima `
  --network_dim=8 `
  --network_alpha=8 `
  --learning_rate=1e-4 `
  --optimizer_type="AdamW8bit" `
  --lr_scheduler="constant" `
  --max_train_steps=500 `
  --mixed_precision="bf16"
```

Dataset-backed iLECO is available from `anima_train_leco.py` with `--ileco_latent_source=dataset`. Internally it reuses the normal Anima dataset, VAE, bucket, and latent caching pipeline:

```powershell
accelerate launch --num_cpu_threads_per_process 1 anima_train_leco.py `
  --ileco_latent_source=dataset `
  --dataset_config="path\to\dataset.toml" `
  --ileco_prompt_pairs="path\to\anima_ileco_pairs.json" `
  --pretrained_model_name_or_path="path\to\anima_model.safetensors" `
  --qwen3="path\to\qwen3.safetensors" `
  --vae="path\to\anima_vae.safetensors" `
  --output_dir="path\to\output_dir" `
  --output_name="anima_ileco_dataset_test" `
  --save_model_as=safetensors `
  --network_module=networks.lora_anima `
  --network_dim=8 `
  --network_alpha=8 `
  --network_train_unet_only `
  --cache_latents `
  --cache_text_encoder_outputs `
  --learning_rate=3e-5 `
  --optimizer_type="AdamW8bit" `
  --lr_scheduler="constant" `
  --max_train_steps=100 `
  --mixed_precision="bf16" `
  --gradient_checkpointing `
  --vae_chunk_size=64 `
  --vae_disable_cache
```

For dataset-backed iLECO, `--gradient_checkpointing` and `--cache_latents` or `--cache_latents_to_disk` are strongly recommended. Without gradient checkpointing, the Anima DiT student forward keeps much more activation memory. Without latent caching, the VAE must stay on GPU during training in addition to the Anima DiT. The iLECO implementation only uses the text encoder to cache prompt-pair conditions, then moves it back to CPU before DiT training starts.

<details>
<summary>日本語</summary>

対応している Anima iLECO には、2種類の latent source があります。

| 実装 | latent の由来 | 挙動 |
|---|---|---|
| `anima_train_leco.py` | 内部の prompt-only latent | prompt-only iLECO |
| `anima_train_leco.py --ileco_latent_source=dataset` | dataset 画像 latent + noise | dataset-backed iLECO |

dataset-backed 経路では、学習画像の内容そのものを semantic target として使っているわけではありません。主な役割は、画像らしい latent、解像度/bucket 挙動、noise を加える基点を提供することです。

prompt-only iLECO が弱い概念では、dataset-backed iLECO 経路を使う方が再現性が高い場合があります。学習画像は完全な target 画像である必要はありませんが、学習したい概念にとって有用な構造を持つ方が有利です。髪色 edit であれば、無関係な風景や抽象画像より、人物やポートレート画像の latent の方が土台として適しています。

prompt-only iLECO は `anima_train_leco.py` で実行できます。dataset-backed iLECO は、通常の dataset、VAE、bucket、latent cache を再利用するため、`anima_train_leco.py --ileco_latent_source=dataset` から呼び出せます。

dataset-backed iLECO では `--gradient_checkpointing` と、`--cache_latents` または `--cache_latents_to_disk` を強く推奨します。gradient checkpointing を使わない場合、Anima DiT の student forward が多くの activation memory を保持します。latent cache を使わない場合、Anima DiT に加えて VAE も学習中 GPU 上に残ります。iLECO 実装では text encoder は prompt pair 条件の cache にだけ使い、その後 DiT 学習が始まる前に CPU へ戻します。

</details>

## Notes / 注意事項

The script precomputes Anima LLM Adapter outputs for prompt conditions by default. Disable it only when intentionally training the LLM Adapter LoRA:

```txt
--no_leco_cache_llm_adapter_outputs
--network_args "train_llm_adapter=true"
```

The following Stable Diffusion-specific options are ignored or not used by Anima iLECO:

```txt
--v2
--v_parameterization
--clip_skip
--zero_terminal_snr
--min_snr_gamma
```

The script is step-based. Use `--max_train_steps` and `--save_every_n_steps`; epoch-based save options are ignored.

`--llm_adapter_path` is currently not loaded separately by the Anima model loader. Use a DiT file that already contains the LLM Adapter weights.

<details>
<summary>日本語</summary>

## 注意事項

prompt 条件の Anima LLM Adapter 出力は既定で事前計算されます。LLM Adapter LoRA を意図的に学習する場合だけ無効にしてください。

```txt
--no_leco_cache_llm_adapter_outputs
--network_args "train_llm_adapter=true"
```

以下の Stable Diffusion 固有オプションは、Anima iLECO では無視されるか使用されません。

```txt
--v2
--v_parameterization
--clip_skip
--zero_terminal_snr
--min_snr_gamma
```

このスクリプトは step ベースです。`--max_train_steps` と `--save_every_n_steps` を使用してください。epoch ベースの保存オプションは無視されます。

`--llm_adapter_path` は現時点では Anima model loader 側で個別読み込みされません。LLM Adapter の重みを含む DiT ファイルを使用してください。

</details>
