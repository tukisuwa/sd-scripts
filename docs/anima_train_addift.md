# Anima ADDifT Training / Anima ADDifT 学習

This document describes ADDifT-style paired-image LoRA training for Anima, using `anima_train_addift.py`.

This is for users who already have an Anima DiT model, Qwen3 text encoder, Anima VAE, and a pair of source/target image folders. The goal is to train a LoRA that applies the visual change shown by the paired images.

<details>
<summary>日本語</summary>

このドキュメントは、`anima_train_addift.py` を使う Anima 向け ADDifT 方式のペア画像 LoRA 学習について説明します。

対象読者は、Anima DiT model、Qwen3 text encoder、Anima VAE、source/target の画像ペアフォルダを用意済みのユーザーです。目的は、画像ペアに含まれる見た目の変化を LoRA として学習することです。

</details>

## Overview / 概要

ADDifT trains an edit LoRA from paired images:

```txt
source image -> target image
```

In this implementation, `conditioning_data_dir` is the source image directory and `image_dir` is the target image directory. Images are paired by filename stem.

The training signal is:

```txt
teacher: LoRA OFF + target image latent + prompt
student: LoRA ON  + source image latent + same prompt
loss:    student prediction -> teacher prediction
```

The teacher and student use the same noise and timestep.

Use this when the desired edit is easier to show with before/after images than with prompt pairs. For prompt-only edits, use `anima_train_leco.py` instead.

<details>
<summary>日本語</summary>

ADDifT はペア画像から edit LoRA を学習します。

```txt
source image -> target image
```

この実装では `conditioning_data_dir` が source image、`image_dir` が target image です。画像はファイル名の stem で対応付けられます。

学習信号は以下です。

```txt
teacher: LoRA OFF + target image latent + prompt
student: LoRA ON  + source image latent + same prompt
loss:    student prediction -> teacher prediction
```

teacher と student は同じ noise と timestep を使います。

プロンプトだけで指定するより、before/after 画像で示す方が分かりやすい edit に向いています。プロンプトだけの edit には `anima_train_leco.py` を使ってください。

</details>

## Requirements / 前提条件

Before running training, prepare these files and folders:

```txt
anima_model.safetensors
qwen3.safetensors
anima_vae.safetensors
addift_dataset.toml
target_images/
source_images/
```

The target and source folders must contain images with matching filename stems:

```txt
target_images/img001.png
source_images/img001.png
target_images/img002.png
source_images/img002.png
```

If an image exists only on one side, dataset preparation fails before training starts. The source and target images should be aligned as much as possible. ADDifT learns image differences, so unrelated pairs can teach unwanted pose, composition, or style changes.

<details>
<summary>日本語</summary>

学習前に以下を用意してください。

```txt
anima_model.safetensors
qwen3.safetensors
anima_vae.safetensors
addift_dataset.toml
target_images/
source_images/
```

target と source の画像は、同じ filename stem で対応付けます。

```txt
target_images/img001.png
source_images/img001.png
target_images/img002.png
source_images/img002.png
```

片方にしか存在しない画像がある場合、学習開始前の dataset 準備で失敗します。source と target はなるべく位置合わせされた画像にしてください。ADDifT は画像差分を学習するため、無関係なペアでは意図しない pose、構図、style の差まで学習しやすくなります。

</details>

## Dataset / データセット

Use a ControlNet-style dataset config with `conditioning_data_dir`:

```toml
[[datasets]]
batch_size = 1
resolution = [512, 512]
enable_bucket = true
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = 'path\to\target_images'
  conditioning_data_dir = 'path\to\source_images'
  num_repeats = 100
  caption_extension = ".txt"
```

Example pairing:

```txt
target_images/img001.png
source_images/img001.png
```

After loading the dataset, `image_dir` is treated as the target side and `conditioning_data_dir` is treated as the source side. This is important because reversing the folders trains the opposite edit.

`batch_size=1` is recommended when using per-pair multipliers, because the LoRA multiplier is global for each forward pass. Per-pair weights work per sample.

<details>
<summary>日本語</summary>

`conditioning_data_dir` を持つ ControlNet 形式の dataset config を使います。

```toml
[[datasets]]
batch_size = 1
resolution = [512, 512]
enable_bucket = true
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = 'path\to\target_images'
  conditioning_data_dir = 'path\to\source_images'
  num_repeats = 100
  caption_extension = ".txt"
```

対応例:

```txt
target_images/img001.png
source_images/img001.png
```

dataset 読み込み後、`image_dir` は target 側、`conditioning_data_dir` は source 側として扱われます。フォルダを逆にすると逆方向の edit を学習するので注意してください。

pair ごとの multiplier を使う場合は `batch_size=1` を推奨します。LoRA multiplier は forward 単位で global に適用されます。pair ごとの weight は sample 単位で機能します。

</details>

## Basic Steps / 基本手順

1. Create the dataset TOML.
2. Run a 1-step smoke test.
3. If the smoke test succeeds, run a short training test such as 100 steps.
4. Load the saved LoRA in inference and compare the same prompt with and without the LoRA.
5. Increase steps or tune options only after the edit direction is visible.

<details>
<summary>日本語</summary>

1. dataset TOML を作成します。
2. まず 1 step の smoke test を実行します。
3. smoke test が成功したら、100 steps などの短い学習を実行します。
4. 保存された LoRA を推論で読み込み、同じ prompt で LoRA あり/なしを比較します。
5. edit 方向が見えてから、steps や option を調整します。

</details>

## Command / コマンド

Start with a short run by setting `--max_train_steps=1`. A successful smoke test should load the dataset, create the LoRA, run one step, and save without errors.

After that, raise `--max_train_steps` to `100` or more for a quick quality check.

```powershell
accelerate launch --num_cpu_threads_per_process 1 anima_train_addift.py `
  --dataset_config="path\to\addift_dataset.toml" `
  --pretrained_model_name_or_path="path\to\anima_model.safetensors" `
  --qwen3="path\to\qwen3.safetensors" `
  --vae="path\to\anima_vae.safetensors" `
  --output_dir="path\to\output_dir" `
  --output_name="anima_addift_test" `
  --save_model_as=safetensors `
  --network_module=networks.lora_anima `
  --network_dim=8 `
  --network_alpha=8 `
  --network_train_unet_only `
  --cache_latents `
  --cache_text_encoder_outputs `
  --addift_cache_conditioning_latents `
  --learning_rate=5e-5 `
  --optimizer_type="AdamW8bit" `
  --lr_scheduler="constant" `
  --max_train_steps=100 `
  --mixed_precision="bf16" `
  --gradient_checkpointing `
  --vae_chunk_size=64 `
  --vae_disable_cache
```

`anima_train_addift.py` enables ADDifT internally, so `--addift` is not required.

The command above caches target latents with `--cache_latents` and source latents with `--addift_cache_conditioning_latents`. This avoids keeping the VAE in the active training path after caching and is the recommended starting point for VRAM usage.

<details>
<summary>日本語</summary>

まずは `--max_train_steps=1` にして短く実行してください。成功時は dataset を読み込み、LoRA を作成し、1 step 実行して、エラーなく保存します。

その後、簡単な品質確認として `--max_train_steps` を `100` 以上に増やします。

`anima_train_addift.py` は内部で ADDifT を有効化するため、`--addift` の指定は不要です。

上の command は `--cache_latents` で target latent を、`--addift_cache_conditioning_latents` で source latent を cache します。cache 後は VAE を active な学習経路に残しにくくなるため、VRAM 使用量の面で最初に試す構成として推奨します。

</details>

## Options / オプション

Useful options:

```txt
--addift_cache_conditioning_latents
--addift_pair_settings
--addift_mask_loss
--addift_mask_data_dir
--addift_alpha_mask
--addift_loss_weight
--addift_multiplier
--addift_min_sigma
--addift_max_sigma
--add_reverse_pairs
--reverse_multiplier
--reverse_weight
```

`--addift_cache_conditioning_latents` caches the source image latents from `conditioning_data_dir`. `--cache_latents` caches only target image latents from `image_dir`.

`--addift_pair_settings` accepts a UTF-8 JSON file keyed by target image filename stem:

```json
{
  "img001": {
    "weight": 1.0,
    "multiplier": 1.0,
    "reverse_weight": 0.5,
    "reverse_multiplier": -1.0
  }
}
```

List form is also accepted:

```json
{
  "pairs": [
    {"key": "img001", "weight": 1.0, "multiplier": 1.0}
  ]
}
```

`--add_reverse_pairs` randomly mixes the reverse direction:

```txt
forward: source image -> target image
reverse: target image -> source image
```

`--addift_min_sigma` and `--addift_max_sigma` restrict the sigma range for ADDifT training.

<details>
<summary>日本語</summary>

`--addift_cache_conditioning_latents` は `conditioning_data_dir` 側の source image latent を cache します。`--cache_latents` が cache するのは `image_dir` 側の target image latent だけです。

`--addift_pair_settings` には、target image の filename stem を key にした UTF-8 JSON を指定できます。

`--add_reverse_pairs` は reverse direction をランダムに混ぜます。

```txt
forward: source image -> target image
reverse: target image -> source image
```

`--addift_min_sigma` と `--addift_max_sigma` は ADDifT 学習で使う sigma 範囲を制限します。

</details>

## Mask Loss / mask loss

To use explicit mask files, set `addift_mask_data_dir` in the dataset TOML and pass `--addift_mask_loss`:

```toml
[[datasets]]
batch_size = 1
resolution = [512, 512]

  [[datasets.subsets]]
  image_dir = 'path\to\target_images'
  conditioning_data_dir = 'path\to\source_images'
  addift_mask_data_dir = 'path\to\loss_masks'
```

Masks are matched by filename stem. White applies loss strongly, black suppresses loss. The mask is resized to latent resolution and multiplied into the ADDifT loss weighting.

Use mask loss when only part of the image should be edited. Do not use it for the first smoke test unless you already know the mask files are present and correctly paired.

You can also derive masks from alpha channels:

```toml
addift_alpha_mask = "target"
```

Supported alpha modes are:

```txt
target
source
union
intersection
difference
```

If both `addift_mask_data_dir` and `addift_alpha_mask` are set, explicit mask files are used first.

<details>
<summary>日本語</summary>

明示的な mask file を使う場合は、dataset TOML に `addift_mask_data_dir` を指定し、`--addift_mask_loss` を付けます。

mask は filename stem で照合されます。白は loss を強く適用し、黒は loss を抑制します。mask は latent 解像度に resize され、ADDifT loss weighting に掛けられます。

画像の一部分だけを edit したい場合に mask loss を使います。mask file が存在し、正しく対応していることを確認済みでない限り、最初の smoke test では使わない方が問題の切り分けが簡単です。

alpha channel から mask を作ることもできます。

```toml
addift_alpha_mask = "target"
```

対応する alpha mode は `target`, `source`, `union`, `intersection`, `difference` です。`addift_mask_data_dir` と `addift_alpha_mask` の両方がある場合は、明示的な mask file が優先されます。

</details>

## Notes / 注意事項

ADDifT learns the difference between paired images. Strongly mismatched pairs can train broad composition or style changes, so start with well-aligned source and target images.

`--gradient_checkpointing`, `--cache_latents`, and `--cache_text_encoder_outputs` are recommended for VRAM usage.

The following modes are not supported together:

```txt
--ileco
--addift
```

<details>
<summary>日本語</summary>

ADDifT はペア画像間の差分を学習します。大きくずれたペアでは構図や絵柄の差まで学習しやすいため、まずは位置合わせされた source / target 画像から試してください。

VRAM 使用量を抑えるため、`--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs` を推奨します。

`--ileco` と `--addift` は同時に使えません。

</details>

## Success and Troubleshooting / 成功基準とトラブルシュート

A minimal run is successful when:

```txt
1 step finishes without an exception
a .safetensors LoRA is saved in output_dir
the LoRA changes inference output in the source -> target direction
```

If dataset loading fails, check that every target image has a source image with the same filename stem.

If mask loss fails, check that every target image has a mask with the same filename stem, or remove `--addift_mask_loss` and test without masks first.

If VRAM usage is too high, use `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`, and `--addift_cache_conditioning_latents`. Reducing `--network_dim` and resolution also helps.

If the LoRA changes pose or composition too much, use more closely aligned source/target pairs, add masks, lower `--addift_multiplier`, or train fewer steps.

If the LoRA effect is weak, verify that `image_dir` is the desired target side and `conditioning_data_dir` is the source side, then try more steps or a slightly higher learning rate.

<details>
<summary>日本語</summary>

最小確認の成功基準は以下です。

```txt
1 step が例外なく終わる
output_dir に .safetensors LoRA が保存される
推論で source -> target 方向の変化が出る
```

dataset 読み込みで失敗する場合は、すべての target image に同じ filename stem の source image があるか確認してください。

mask loss で失敗する場合は、すべての target image に同じ filename stem の mask があるか確認してください。切り分ける場合は `--addift_mask_loss` を外して、まず mask なしで確認します。

VRAM 使用量が大きい場合は、`--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`, `--addift_cache_conditioning_latents` を使ってください。`--network_dim` や resolution を下げることも有効です。

LoRA が pose や構図まで変えすぎる場合は、source/target の位置合わせを強める、mask を使う、`--addift_multiplier` を下げる、steps を減らす、などを試してください。

LoRA 効果が弱い場合は、`image_dir` が target 側、`conditioning_data_dir` が source 側になっているか確認してから、steps を増やすか learning rate を少し上げてください。

</details>
