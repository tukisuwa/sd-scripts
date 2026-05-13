# Architecture-agnostic LoRA merger / アーキテクチャ非依存 LoRA マージャ

`merger.py` merges multiple sd-scripts style LoRA files into a new LoRA file by reconstructing each LoRA delta weight and decomposing the merged delta with SVD.

`merger.py` は、複数の sd-scripts 形式 LoRA を合成し、各 LoRA の差分重みを再構成したうえで SVD により新しい LoRA として再分解するツールです。

## Positioning / 位置づけ

This script does not load a base model. It operates only on LoRA state dict keys such as:

このスクリプトはベースモデルを読み込みません。次のような LoRA の state dict キーだけを対象に処理します。

- `*.lora_down.weight`
- `*.lora_up.weight`
- `*.alpha`

Because of that, it can be used without directly depending on whether the original training model was SD 1.x, SDXL, SD3, FLUX.1, Anima, or another architecture, as long as the LoRA files use compatible sd-scripts style keys and tensor shapes.

そのため、LoRA ファイルが互換性のある sd-scripts 形式のキーと tensor shape を持っていれば、学習元が SD 1.x、SDXL、SD3、FLUX.1、Anima、その他のアーキテクチャであるかを直接は要求しません。

It is best described as:

実態としては次のようなツールです。

**Architecture-agnostic SVD merger for sd-scripts style LoRA state dicts.**

**sd-scripts 形式 LoRA state dict 向けの、アーキテクチャ非依存 SVD マージャです。**

## Requirements / 前提

- Run it from the repository root, where `library/` is importable.
- `safetensors`, `torch`, and the sd-scripts local `library` package must be available.
- Input LoRAs should use sd-scripts style `lora_down` / `lora_up` keys.
- LoRAs should normally come from compatible target architectures. The script is base-model independent, but it cannot make incompatible module shapes compatible.

- `library/` を import できるリポジトリ直下から実行してください。
- `safetensors`、`torch`、sd-scripts のローカル `library` package が必要です。
- 入力 LoRA は sd-scripts 形式の `lora_down` / `lora_up` キーを持つ必要があります。
- 通常は互換性のある対象アーキテクチャ由来の LoRA 同士で使ってください。ベースモデル非依存ではありますが、互換性のない module shape を自動的に合わせるものではありません。

## Basic Usage / 基本的な使い方

```bash
python merger.py \
  --models lora_a.safetensors lora_b.safetensors \
  --ratios 0.7 0.3 \
  --new_rank 16 \
  --save_to merged_lora.safetensors
```

For 3x3 convolution LoRA layers, specify a separate rank:

3x3 convolution LoRA 層に別 rank を使う場合:

```bash
python merger.py \
  --models lora_a.safetensors lora_b.safetensors \
  --ratios 1.0 0.5 \
  --new_rank 16 \
  --new_conv_rank 8 \
  --save_to merged_lora.safetensors
```

Use `--device cuda` to run the SVD computation on GPU:

SVD 計算を GPU で行う場合:

```bash
python merger.py \
  --models lora_a.safetensors lora_b.safetensors \
  --ratios 1.0 1.0 \
  --new_rank 16 \
  --device cuda \
  --save_to merged_lora.safetensors
```

## SVD Modes / SVD モード

`--svd_mode quality`

Uses full SVD. This is the most conservative mode and is the default.

full SVD を使います。最も保守的なモードで、デフォルトです。

`--svd_mode speed`

Uses randomized low-rank SVD with:

次の設定で randomized low-rank SVD を使います。

```text
q = new_rank + svd_oversample
```

This is faster for experimentation, but may have more reconstruction error than full SVD.

試行錯誤では高速ですが、full SVD より再構成誤差が大きくなる場合があります。

`--svd_mode resize_lora`

Follows the approach used by `networks/resize_lora.py`:

`networks/resize_lora.py` に近い方針です。

```text
if out_size > 2048 and in_size > 2048 and svd_niter > 0:
    use svd_lowrank with q = min(2 * rank, out_size, in_size)
else:
    use full SVD
```

This is usually a better speed/quality compromise than `speed` mode.

通常は `speed` モードより速度と品質のバランスを取りやすいです。

## Clamping / clamp

By default, SVD results are clamped to a quantile range to suppress extreme values.

デフォルトでは、SVD 結果の極端な値を抑えるため quantile 範囲で clamp します。

Use `--no_clamp` to disable this behavior:

無効化する場合は `--no_clamp` を指定します。

```bash
python merger.py \
  --models lora_a.safetensors lora_b.safetensors \
  --ratios 1.0 1.0 \
  --new_rank 16 \
  --no_clamp \
  --save_to merged_lora.safetensors
```

`--no_clamp` is independent from `--svd_mode`. It applies to `quality`, `speed`, and `resize_lora`.

`--no_clamp` は `--svd_mode` とは独立しています。`quality`、`speed`、`resize_lora` のすべてに適用されます。

## Module Filtering / モジュールフィルタ

Use `--module_filter` to include or exclude modules by regular expression.

`--module_filter` で正規表現によりモジュールを include / exclude できます。

Format:

形式:

```text
PATTERN:ACTION[:STRENGTH]
```

- `ACTION` is `include` or `exclude`.
- `STRENGTH` is used only with `include`.
- Rules are applied in order. Later rules can override earlier rules.

- `ACTION` は `include` または `exclude` です。
- `STRENGTH` は `include` の場合だけ使われます。
- ルールは指定順に適用されます。後のルールで前のルールを上書きできます。

Example:

例:

```bash
python merger.py \
  --models style.safetensors character.safetensors \
  --ratios 1.0 1.0 \
  --new_rank 16 \
  --module_filter ".*attn.*:include:0.8" ".*text_encoder.*:exclude" \
  --save_to merged_filtered.safetensors
```

Use `--filter_by_first_lora` to initially include only modules that exist in the first LoRA. `--module_filter` rules are applied after that.

`--filter_by_first_lora` を指定すると、最初の LoRA に存在するモジュールだけを初期対象にします。その後に `--module_filter` が適用されます。

## Quantization Before SVD / SVD 前の量子化

`--quantize` applies an experimental quantization-like transform to the merged delta weight before SVD.

`--quantize` は、SVD 前の合成済み差分重みに実験的な量子化風の変換を適用します。

Available methods:

利用可能な方式:

- `uniform`
- `turbo`
- `rotor`

`turbo` and `rotor` are approximations for experimentation. They are not full paper implementations.

`turbo` と `rotor` は実験用の近似実装です。論文の完全な実装ではありません。

## Metadata and Hashes / メタデータと hash

The output metadata includes merge parameters such as ratios, rank, SVD mode, quantization settings, and module filters.

出力メタデータには、ratio、rank、SVD mode、量子化設定、module filter などのマージ条件が保存されます。

`sshs_model_hash` and `sshs_legacy_hash` are calculated with `library.train_util.precalculate_safetensors_hashes`, matching the approach used by sd-scripts tools such as `networks/svd_merge_lora.py` and `networks/resize_lora.py`.

`sshs_model_hash` と `sshs_legacy_hash` は、`networks/svd_merge_lora.py` や `networks/resize_lora.py` と同様に `library.train_util.precalculate_safetensors_hashes` で計算されます。

## Difference from Existing Tools / 既存ツールとの違い

`networks/merge_lora.py`

Merges LoRA files directly, or concatenates ranks with `--concat`. It does not focus on SVD recomposition to a target rank.

LoRA を直接合成、または `--concat` で rank 結合します。指定 rank への SVD 再構成が主目的ではありません。

`networks/svd_merge_lora.py`

Merges multiple LoRAs and recomposes them with SVD. It is close in purpose, but is more tied to existing sd-scripts layer-weight behavior and does not provide regex module filtering or quantization modes.

複数 LoRA を合成し SVD で再構成します。用途は近いですが、既存 sd-scripts の層別重み指定に寄っており、正規表現 module filter や量子化モードはありません。

`networks/resize_lora.py`

Changes the rank of a single LoRA. `merger.py --svd_mode resize_lora` borrows its low-rank SVD strategy, but `merger.py` first merges multiple LoRAs.

単一 LoRA の rank を変更します。`merger.py --svd_mode resize_lora` はこの低ランク SVD 方針に寄せていますが、`merger.py` は先に複数 LoRA を合成します。

## Naming Note / 名称について

`merger.py` is short, but can be confused with existing tools such as `networks/merge_lora.py` and `networks/svd_merge_lora.py`.

`merger.py` は短い名前ですが、既存の `networks/merge_lora.py` や `networks/svd_merge_lora.py` と混同しやすい名前でもあります。

If the script is kept at the repository root, a more descriptive future name could be:

リポジトリ直下に置く前提で、将来的により説明的な名前にするなら次の候補があります。

- `generic_lora_merger.py`
- `lora_svd_merger.py`
- `state_dict_lora_merger.py`

For compatibility with existing local workflows, keeping `merger.py` is reasonable if this document and the command help make its scope clear.

既存のローカル運用との互換性を優先するなら、ドキュメントと help で用途を明確にしたうえで `merger.py` のまま維持するのは妥当です。
