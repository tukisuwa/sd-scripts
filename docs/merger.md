# Architecture-agnostic LoRA merger / アーキテクチャ非依存 LoRA マージツール

`merger.py` merges multiple sd-scripts style LoRA files into a new LoRA file by reconstructing each LoRA delta weight and decomposing the merged delta with SVD.

`merger.py` は、複数の sd-scripts 形式 LoRA を合成し、各 LoRA の差分重みを再構成したうえで SVD により新しい LoRA として再分解するツールです。

This document is for users who already have two or more LoRA files and want to create one new LoRA file from them without loading the base model.

このドキュメントは、複数の LoRA ファイルを持っていて、ベースモデルを読み込まずに 1 つの新しい LoRA ファイルへ合成したいユーザー向けです。

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

**sd-scripts 形式 LoRA state dict 向けの、アーキテクチャ非依存 SVD マージツールです。**

Use this tool when you want a target-rank LoRA output after merging. If you only need to merge LoRA weights directly into a Stable Diffusion checkpoint, existing model-specific merge scripts may be more appropriate.

複数 LoRA を合成したあと、指定 rank の LoRA として出力したい場合に使います。Stable Diffusion checkpoint に LoRA 重みを直接マージしたいだけなら、既存のモデル別 merge script の方が適している場合があります。

## Requirements / 前提

- Run it from the repository root, where `library/` is importable.
- `safetensors`, `torch`, and the sd-scripts local `library` package must be available.
- Input LoRAs should use sd-scripts style `lora_down` / `lora_up` keys.
- LoRAs should normally come from compatible target architectures. The script is base-model independent, but it cannot make incompatible module shapes compatible.

- `library/` を import できるリポジトリ直下から実行してください。
- `safetensors`、`torch`、sd-scripts のローカル `library` package が必要です。
- 入力 LoRA は sd-scripts 形式の `lora_down` / `lora_up` キーを持つ必要があります。
- 通常は互換性のある対象アーキテクチャ由来の LoRA 同士で使ってください。ベースモデル非依存ではありますが、互換性のない module shape を自動的に合わせるものではありません。

## Before You Start / 実行前に確認すること

Check these items before running the command:

1. Decide the input LoRA files and their order.
2. Decide the ratio for each LoRA in the same order as `--models`.
3. Decide the output rank with `--new_rank`.
4. Decide the output filename with `--save_to`.

The number of `--models` entries must match the number of `--ratios` entries.

Start with `--new_rank` equal to the rank you normally use for LoRA inference. If the result loses too much detail, try a larger rank. If the file is too large, try a smaller rank.

<details>
<summary>日本語</summary>

実行前に以下を確認してください。

1. 入力する LoRA ファイルと、その順番を決めます。
2. `--models` と同じ順番で、各 LoRA の ratio を決めます。
3. `--new_rank` で出力 LoRA の rank を決めます。
4. `--save_to` で出力ファイル名を決めます。

`--models` の個数と `--ratios` の個数は一致している必要があります。

まずは普段の LoRA 推論で使いやすい rank を `--new_rank` に指定してください。結果の情報量が足りない場合は rank を上げ、ファイルサイズを小さくしたい場合は rank を下げます。

</details>

## Basic Usage / 基本的な使い方

Run this from the repository root. The command creates `merged_lora.safetensors`.

リポジトリ直下から実行します。成功すると `merged_lora.safetensors` が作成されます。

```bash
python merger.py \
  --models lora_a.safetensors lora_b.safetensors \
  --ratios 0.7 0.3 \
  --new_rank 16 \
  --save_to merged_lora.safetensors
```

In this example, `lora_a` contributes `0.7` and `lora_b` contributes `0.3` to the merged delta. Ratios are not required to add up to `1.0`; they are multipliers.

この例では、合成差分に対して `lora_a` を `0.7`、`lora_b` を `0.3` の強さで使います。ratio は合計 `1.0` にする必要はなく、各 LoRA への倍率として扱われます。

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

SVD recomposes the merged full delta back into LoRA down/up matrices. The mode controls the speed and reconstruction quality of that decomposition.

SVD は、合成済みの full delta を LoRA の down/up 行列へ再分解する処理です。mode は、その分解の速度と再構成品質を切り替えます。

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

Keep clamping enabled at first. Disable it only when you specifically want to preserve large SVD values and are willing to check the output for artifacts.

最初は clamp を有効のまま使ってください。大きな SVD 値を意図的に残したい場合だけ `--no_clamp` を使い、出力に破綻がないか確認してください。

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

Use filtering when you want to merge only part of a LoRA, such as attention modules, or when one LoRA contains modules you do not want in the output.

attention module だけを合成したい場合や、出力に含めたくない module がある LoRA を扱う場合に filter を使います。

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

Do not enable quantization for your first merge. Use it only when you are comparing experimental variants and can inspect the generated output.

最初の merge では量子化を有効にしないでください。生成結果を比較できる状態で、実験的な variant を試す場合だけ使ってください。

## Metadata and Hashes / メタデータと hash

The output metadata includes merge parameters such as ratios, rank, SVD mode, quantization settings, and module filters.

出力メタデータには、ratio、rank、SVD mode、量子化設定、module filter などのマージ条件が保存されます。

`sshs_model_hash` and `sshs_legacy_hash` are calculated with `library.train_util.precalculate_safetensors_hashes`, matching the approach used by sd-scripts tools such as `networks/svd_merge_lora.py` and `networks/resize_lora.py`.

`sshs_model_hash` と `sshs_legacy_hash` は、`networks/svd_merge_lora.py` や `networks/resize_lora.py` と同様に `library.train_util.precalculate_safetensors_hashes` で計算されます。

## Success and Troubleshooting / 成功基準とトラブルシュート

A successful run should:

```txt
finish without an exception
write the file specified by --save_to
print progress for processed LoRA modules
save metadata describing ratios, rank, SVD mode, and filters
```

After merging, load the output LoRA in your inference workflow and compare it against the input LoRAs. The expected result is an approximation of the weighted combination, not a perfect copy.

If the command fails because `--models` and `--ratios` lengths differ, add or remove ratios so both lists have the same length.

If it fails with missing or incompatible module shapes, the input LoRAs are probably from incompatible targets or different module layouts. Try LoRAs trained for the same architecture and network module.

If the merged LoRA is weak or loses detail, increase `--new_rank` or use `--svd_mode quality`.

If the output is unstable or produces artifacts, lower ratios, keep clamping enabled, avoid quantization, or try a smaller `--new_rank`.

If the merge is slow, try `--device cuda` or `--svd_mode resize_lora`. Use `speed` mode mainly for quick experiments.

<details>
<summary>日本語</summary>

成功時は以下の状態になります。

```txt
例外なく終了する
--save_to で指定したファイルが作成される
処理された LoRA module の進捗が表示される
ratio、rank、SVD mode、filter などの metadata が保存される
```

merge 後は、出力 LoRA を普段の推論環境で読み込み、入力 LoRA と比較してください。期待される結果は、重み付き合成の近似であり、完全なコピーではありません。

`--models` と `--ratios` の数が違うというエラーが出る場合は、両方の数が同じになるように ratio を追加または削除してください。

module shape の不足や不一致で失敗する場合、入力 LoRA の対象モデルや module 構成が互換でない可能性があります。同じ architecture と network module で学習した LoRA 同士を試してください。

merge 後の LoRA が弱い、または情報量が落ちる場合は、`--new_rank` を上げるか `--svd_mode quality` を使ってください。

出力が不安定、または破綻する場合は、ratio を下げる、clamp を有効のままにする、量子化を使わない、`--new_rank` を小さくする、などを試してください。

処理が遅い場合は、`--device cuda` または `--svd_mode resize_lora` を試してください。`speed` mode は主に短い実験用です。

</details>

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
