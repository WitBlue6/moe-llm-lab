# MoE LLM Lab：按 epoch 训练速查

2026-09-29；单机 8×3090。各阶段先以 **1 epoch** 建立实验，不保证一轮达到可用质量。

**终端显示：** 同步最新版 `src/moe_llm/` 后，下列文本、视觉适配和 SigLIP 训练命令自动显示底部进度条，无需新增参数。八卡只由 rank 0 显示，例如 `Epoch 1/2 [###---------] 250/1000 train=3.2100 val=3.4800@200`；上方持续打印每步 JSON 指标。`val` 是最近一次验证，`@200` 是其全局步号，首次验证前为 `--`，仍按配置 `eval_every` 及本次结束时验证。验证/保存期间显示 `validate` / `save`。

重定向日志时自动关闭动态显示；`MOE_LAB_PROGRESS=0` 手动关闭，`MOE_LAB_PROGRESS=1` 强制显示（经 tee 写入文件时也会留下控制字符）。本次代码更新改变严格恢复的指纹，正在跑或需要精确恢复的旧实验继续用旧代码；不要为了显示进度条绕过恢复校验。

沿用已有 `data/tokenizer-v1.json`、`configs/model-text-v1.json`、`data/pretrain-v1`、`data/sft-v1`。已完成数据准备者从第 1 步开始；不要重新训练 tokenizer。这里是新实验，默认从随机文本权重开始，不自动续接旧的 1000/500-step 实验。

所有命令从服务器 `~/moe-llm-lab` 执行。保持同一个 shell；重连后重设变量和函数。新输出不得覆盖旧目录。命令出错时先停下处理，不继续下一阶段。

## 0. 仅尚未准备文本数据时执行

```bash
cd ~/moe-llm-lab
uv sync --locked --extra vision
uv run --locked python scripts/download_minimind.py --preset mini --output data/downloads/minimind-mini-v1
uv run --locked python scripts/convert_minimind.py \
  --pretrain data/downloads/minimind-mini-v1/pretrain_t2t_mini.jsonl \
  --sft data/downloads/minimind-mini-v1/sft_t2t_mini.jsonl --output data/raw/minimind-mini-v1
uv run --locked moe-lab tokenizer --input data/raw/minimind-mini-v1/pretrain.jsonl \
  --output data/tokenizer-v1.json --vocab-size 16384 --val-ratio 0.05 --seed 42
uv run --locked moe-lab configure --model-config configs/moe-base.json \
  --tokenizer data/tokenizer-v1.json --output configs/model-text-v1.json
uv run --locked moe-lab prepare --input data/raw/minimind-mini-v1/pretrain.jsonl \
  --output data/pretrain-v1 --tokenizer data/tokenizer-v1.json --stage pretrain \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
uv run --locked moe-lab prepare --input data/raw/minimind-mini-v1/sft.jsonl \
  --output data/sft-v1 --tokenizer data/tokenizer-v1.json --stage sft \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
```

## 1. 环境、实验名与 checkpoint 函数

```bash
cd ~/moe-llm-lab
uv sync --locked --extra vision
uv run --locked moe-lab doctor
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN=epoch1-20260929
export EPOCHS=1
export TOK=data/tokenizer-v1.json
export MC=configs/model-text-v1.json
export CFG=configs/$RUN
mkdir -p reports

# 只取已完成指定轮数的 run 的最后 checkpoint；不是自动挑选质量最好的 checkpoint。
final_ckpt() {
  uv run --locked python - "$1" "$EPOCHS" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / 'summary.json').read_text())
assert m.get('epochs_completed', 0) >= int(sys.argv[2]) - 1e-8, '该 run 尚未完成目标轮数'
p = root / f"step-{int(m['step']):07d}.pt"
assert p.is_file(), f'checkpoint 不存在: {p}'
print(p)
PY
}
```

## 2. 创建本次固定配置（只执行一次）

模板学习率仅作为起点。这里将文本/VLM 验证及保存间隔设为 500 步，减少整轮训练的验证与磁盘开销；warmup 沿用模板。启动提示 warmup 不小于总步数时，在开始正式 run 前缩短新配置的 warmup。

```bash
uv run --locked python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['CFG'])
root.mkdir(parents=True, exist_ok=False)
for source, target in [('train-pretrain.json', 'pretrain.json'),
                       ('train-sft.json', 'sft.json'),
                       ('train-vision-align.json', 'align.json'),
                       ('train-vision-sft.json', 'vision-sft.json'),
                       ('siglip-scratch.json', 'siglip.json')]:
    c = json.loads((Path('configs') / source).read_text())
    c['epochs'] = int(os.environ['EPOCHS'])
    if source != 'siglip-scratch.json':
        c['eval_every'] = 500
        c['save_every'] = 500
    with (root / target).open('x') as f:
        json.dump(c, f, indent=2)
PY
```

## 3. 文本单卡与八卡短跑

保存成功、loss/梯度有限后再正式跑。短跑输出不作为正式阶段的输入。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision moe-lab train \
  --model-config "$MC" --train-config "$CFG/pretrain.json" \
  --data data/pretrain-v1 --tokenizer "$TOK" \
  --stop-after 9 --output "runs/$RUN-text-pilot-1gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config "$MC" --train-config "$CFG/pretrain.json" \
  --data data/pretrain-v1 --tokenizer "$TOK" \
  --stop-after 10 --output "runs/$RUN-text-pilot-8gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

## 4. 文本预训练一轮

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config "$MC" --train-config "$CFG/pretrain.json" \
  --data data/pretrain-v1 --tokenizer "$TOK" \
  --output "runs/$RUN-pretrain" \
  --epochs "$EPOCHS" --eval-max-batches 128
```

```bash
PRETRAIN=$(final_ckpt "runs/$RUN-pretrain")
uv run --locked moe-lab generate --checkpoint "$PRETRAIN" --tokenizer "$TOK" \
  --prompt '机器学习是一种' --max-new-tokens 128 --temperature 0.8
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab evaluate --checkpoint "$PRETRAIN" \
  --data data/pretrain-v1 --tokenizer "$TOK" --device cuda --batch-size 1
```

全量评测可能较久。用固定多个续写样例检查质量；仍严重重复时先处理文本基座，不急着扩展视觉。按历史 N=1,286,338、8 卡、batch1、累积8，一轮为 20,100 steps；实际看 `training_budget`。

## 5. 文本 SFT 一轮

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision moe-lab train \
  --model-config "$MC" --train-config "$CFG/sft.json" \
  --data data/sft-v1 --tokenizer "$TOK" \
  --init-from "$PRETRAIN" --stop-after 9 --output "runs/$RUN-sft-pilot-1gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config "$MC" --train-config "$CFG/sft.json" \
  --data data/sft-v1 --tokenizer "$TOK" \
  --init-from "$PRETRAIN" --output "runs/$RUN-sft" \
  --epochs "$EPOCHS" --eval-max-batches 128
```

```bash
BASE=$(final_ckpt "runs/$RUN-sft")
uv run --locked moe-lab generate --checkpoint "$BASE" --tokenizer "$TOK" --chat \
  --prompt '请用三句话解释为什么需要训练集和验证集。' --max-new-tokens 160 --temperature 0
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab evaluate --checkpoint "$BASE" \
  --data data/sft-v1 --tokenizer "$TOK" --device cuda --batch-size 1
```

历史 N=701,927 对应 10,968 steps。验收多个固定问答后固定 `BASE`；下列全部视觉阶段均使用同一份文本 SFT。若已有合格基座，可跳过第 3–5 步，手动设置 `BASE` 为实际存在的文本 SFT checkpoint。

## 6. 视觉数据（已有相同产物则跳过）

```bash
uv run --locked --extra vision python scripts/download_visual_data.py \
  --stage both --output data/downloads/minimind-vision-v1
uv run --locked --extra vision python scripts/convert_visual_data.py \
  --pretrain data/downloads/minimind-vision-v1/pretrain_i2t.parquet \
  --sft data/downloads/minimind-vision-v1/sft_i2t.parquet --output data/raw/vision-v1
```

来源为 MiniMind-V 聚合数据，脚本固定版本并校验 SHA256；其 ALLaVA 上游含非商业使用限制，许可证说明见原 training-guide 第 10 步。

## 7. 视觉权重：A / B 只选一条

### A. 加载已有权重

已有相同模型文件时跳过下载。选择后跳到第 8 步。

```bash
uv run --locked --extra vision python scripts/download_siglip_weights.py \
  --output models/siglip-base-patch16-224
export VISION_ROUTE=pretrained
```

### B. 随机初始化训练 SigLIP 式双编码器

已有相同 pairs 数据则跳过 prepare。一轮仅是首轮预算，检索验证差时不要直接当作合格视觉骨干。

```bash
uv run --locked --extra vision python -m moe_llm.siglip_training prepare \
  --input data/raw/vision-v1/captions.jsonl --image-root data/raw/vision-v1/images \
  --output data/siglip-pairs-v1 --tokenizer "$TOK" --text-length 128 --val-ratio 0.05 --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision python -m moe_llm.siglip_training train \
  --config "$CFG/siglip.json" --data data/siglip-pairs-v1 \
  --tokenizer "$TOK" --epochs "$EPOCHS" \
  --output "runs/$RUN-siglip-pilot-1gpu" --stop-after 10
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.siglip_training train \
  --config "$CFG/siglip.json" --data data/siglip-pairs-v1 \
  --tokenizer "$TOK" --epochs "$EPOCHS" \
  --output "runs/$RUN-siglip-pilot-8gpu" --stop-after 10
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.siglip_training train \
  --config "$CFG/siglip.json" --data data/siglip-pairs-v1 \
  --tokenizer "$TOK" --epochs "$EPOCHS" \
  --output "runs/$RUN-siglip"
```

```bash
SIGLIP=$(final_ckpt "runs/$RUN-siglip")
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision python -m moe_llm.siglip_training evaluate \
  --checkpoint "$SIGLIP" --data data/siglip-pairs-v1 --device cuda --batch-size 16 --max-samples 256
```

对照早期 checkpoint 的双向 Recall@1、loss 与实际检索样例后，再导出。256 候选的随机 R@1 为 1/256。此入口使用配置 `eval_samples`，不接受 `--eval-max-batches`。

```bash
uv run --locked --extra vision python -m moe_llm.siglip_training export \
  --checkpoint "$SIGLIP" --output "models/$RUN-siglip"
export VISION_ROUTE=scratch
```

## 8. 固定视觉配置与预处理

选定路线后不要原地切换编码器；不同路线须分别训练 projector/LoRA。

```bash
uv run --locked python - <<'PY'
import json, os
from pathlib import Path
route = os.environ['VISION_ROUTE']
assert route in ('pretrained', 'scratch')
source = 'vision-siglip.json' if route == 'pretrained' else 'vision-siglip-scratch.json'
c = json.loads((Path('configs') / source).read_text())
if route == 'scratch':
    c['encoder_path'] = f"models/{os.environ['RUN']}-siglip"
with (Path(os.environ['CFG']) / 'vision.json').open('x') as f:
    json.dump(c, f, indent=2)
PY
export VC=$CFG/vision.json
export VA=data/$RUN-vision-align
export VS=data/$RUN-vision-sft
uv run --locked --extra vision moe-lab vision-prepare \
  --input data/raw/vision-v1/vision-align.jsonl --image-root data/raw/vision-v1/images \
  --output "$VA" --tokenizer "$TOK" --vision-config "$VC" \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
uv run --locked --extra vision moe-lab vision-prepare \
  --input data/raw/vision-v1/vision-sft.jsonl --image-root data/raw/vision-v1/images \
  --output "$VS" --tokenizer "$TOK" --vision-config "$VC" \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
```

检查两个 manifest 的保留条数和超长过滤数量；512 包含 64 个视觉位置。若多数被过滤，先新建更长窗口数据并重新显存短跑，不直接更改进行中的 run。

## 9. 视觉对齐一轮：只训练 projector

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision moe-lab vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/align.json" \
  --data "$VA" --tokenizer "$TOK" \
  --stop-after 9 --output "runs/$RUN-align-pilot-1gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/align.json" \
  --data "$VA" --tokenizer "$TOK" \
  --stop-after 10 --output "runs/$RUN-align-pilot-8gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/align.json" \
  --data "$VA" --tokenizer "$TOK" \
  --output "runs/$RUN-align" \
  --epochs "$EPOCHS" --eval-max-batches 128
```

```bash
ALIGN=$(final_ckpt "runs/$RUN-align")
```

## 10. 视觉 SFT 一轮：projector＋视觉 LoRA

第一次启动使用 `--init-from "$ALIGN"`。**这里不执行 resume。**

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision moe-lab vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/vision-sft.json" \
  --data "$VS" --tokenizer "$TOK" \
  --init-from "$ALIGN" --stop-after 9 --output "runs/$RUN-vision-sft-pilot-1gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/vision-sft.json" \
  --data "$VS" --tokenizer "$TOK" \
  --init-from "$ALIGN" --stop-after 10 --output "runs/$RUN-vision-sft-pilot-8gpu" \
  --epochs "$EPOCHS" --eval-max-batches 9
```

```bash
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/vision-sft.json" \
  --data "$VS" --tokenizer "$TOK" \
  --init-from "$ALIGN" --output "runs/$RUN-vision-sft" \
  --epochs "$EPOCHS" --eval-max-batches 128
```

```bash
VLM=$(final_ckpt "runs/$RUN-vision-sft")
```

## 11. 独立全量验收与生成

以下评测包含正常图像、置零图像和全量文本 logits 比较，耗时可能较长；不继承训练的验证上限。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision moe-lab vision-evaluate \
  --base-checkpoint "$BASE" --checkpoint "$VLM" --tokenizer "$TOK" \
  --data "$VS" --text-data data/sft-v1 --zero-images \
  --device cuda --batch-size 1 --output "reports/$RUN-vision-eval.json"

VLM_TEST_IMAGE=$(uv run --locked python - "$VS" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / 'manifest.json').read_text())
with (root / 'val.jsonl').open() as f:
    row = json.loads(next(f))
print(Path(m['image_root']) / row['image'])
PY
)
uv run --locked --extra vision moe-lab vision-generate \
  --base-checkpoint "$BASE" --checkpoint "$VLM" --tokenizer "$TOK" \
  --image "$VLM_TEST_IMAGE" --prompt '描述这张图片。' --temperature 0 --max-new-tokens 128
uv run --locked --extra vision moe-lab vision-generate \
  --base-checkpoint "$BASE" --checkpoint "$VLM" --tokenizer "$TOK" \
  --prompt '请解释梯度下降。' --temperature 0 --max-new-tokens 128
```

检查 `text_logits_identical`、`text_max_logit_error`，并做多张留出图片的同问换图检查。单张图片和 CE 不能代表完整视觉质量。

## 12. 可选：仅在中断后恢复，不是下一训练阶段

先找到真实存在的 checkpoint；如果没有保存成功的文件，只能重新开始该阶段。下面以八卡视觉 SFT 为例。

```bash
ls -lh "runs/$RUN-vision-sft"/step-*.pt
# 将下行替换成上面实际存在、且未到目标总步数的文件。
RESUME_CKPT="runs/$RUN-vision-sft/step-0000500.pt"
test -f "$RESUME_CKPT" && \
uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint "$BASE" --vision-config "$VC" --train-config "$CFG/vision-sft.json" \
  --data "$VS" --tokenizer "$TOK" \
  --resume "$RESUME_CKPT" --output "runs/$RUN-vision-sft-resumed" \
  --epochs "$EPOCHS" --eval-max-batches 128
```

恢复时保留 `EPOCHS`、配置、代码、数据、基座、编码器、GPU 数和验证上限；移除 `--init-from`，写新目录。恢复完成后执行：

```bash
VLM=$(final_ckpt "runs/$RUN-vision-sft-resumed")
```

再执行第 11 步；若报告已存在则换报告名。文本/对齐恢复同理：使用相同阶段的配置与数据，以 `--resume` 替代 `--init-from`（如有），换新输出目录。SigLIP 恢复在对应 train 命令中添加 `--resume`，同样保留原 epochs 和配置。

## 参数速记

- `--epochs`：正整数，总轮数，覆盖 JSON 的 `max_steps`；不是 `--epoches`。
- `--stop-after`：绝对 optimizer step，提前停止，不改变整轮 LR 计划；必须不大于总预算。
- 文本/VLM 每轮 steps：`ceil(ceil(ceil(N/W)/B)/A)`；DDP 可能补少量重复样本，累积尾批单独更新。
- SigLIP 每轮 steps：`floor(N/(W×B))`，丢尾批、无累积。
- `--eval-max-batches 128`：当前八卡 batch1 最多验证固定前 1024 条；不是训练步数，也不是随机抽样。
- 独立 evaluate/vision-evaluate 是全量评测；SigLIP 使用 `eval_samples`/`--max-samples`。
- `BASE` 固定文本 SFT；`ALIGN` 是视觉对齐 adapter；`VLM` 是视觉 SFT adapter，不能单独替代完整 LLM。
- `final_ckpt` 仅避免硬编码步号；若验证显示中间 checkpoint 更好，手动修改相应变量，并保持后续视觉阶段基座一致。
