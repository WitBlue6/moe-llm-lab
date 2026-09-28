# MoE LLM Lab

以**完整掌握流程**为目标，用原生 PyTorch 搭建小型 MoE 语言模型。

已实现文本网络、tokenizer、预训练、全参数 SFT、评测、生成、恢复和 DDP；现已加入手写 SigLIP 视觉网络（可加载外部权重，或用图文对从零训练）、冻结视觉基座、投影层、视觉 LoRA、视觉对齐/视觉 SFT、图文生成与文本保留检查。DPO、奖励模型、PPO、GRPO 和独立文本 LoRA 微调仍为后续阶段。

**开始正式训练请先读 [逐步训练指南](docs/training-guide.md)**：先完成第 0–8 步文本训练，再进行视觉训练。

视觉两条路线、数据下载与转换、从零训练及 VLM 接入命令见训练指南第 9–13 步。小规模 SigLIP 式预训练不是官方训练配方/效果复现；当前新增链路仅完成工程测试。

## 模型规模

文本、视觉适配和 SigLIP 对比训练都支持 `--epochs N`（或训练 JSON 的 `epochs`），按实际数据量和并行配置计算总步数及学习率计划；未指定则继续使用 `max_steps`。CLI 轮数优先，`--stop-after` 只提前停止。恢复时保留原轮数与训练配置，日志显示 `epochs_completed`。尾批处理与完整用法见训练指南第 5.0 步。

| 配置 | 总参数 | 每 token 激活参数 | 用途 |
| --- | ---: | ---: | --- |
| `configs/moe-base.json` | 264,315,648 | 151,069,440 | 8 卡 3090 的主实验起点 |
| `configs/moe-small.json` | 48,259,584 | 29,385,216 | 保留的较小对照配置 |
| `configs/moe-tiny.json` | 约 5 万 | 由 inspect 输出 | 本地合成数据测试，不代表模型能力 |

主配置：16 层、hidden 768、GQA 12/4 heads、4 个 SwiGLU 专家、top-2 路由、16K 词表、最大上下文 2048。支持 Dense FFN（`num_experts=experts_per_token=1`）、RoPE、RMSNorm、共享 embedding、负载均衡损失及 activation checkpointing。

激活参数统计包含完整 embedding 表，与 FLOPs、显存和实测速度不同。主配置只是建议起点，**尚未在用户的 8 卡 3090 上训练或测量显存**。未来 PPO 同时涉及多个模型和 rollout，保留小 batch、梯度累积和缩短序列的余地。

## 安装与测试

```bash
uv sync --locked
uv run --locked moe-lab doctor
uv run --locked moe-lab inspect --model-config configs/moe-base.json
uv run --locked pytest -q
```

Python 由 `.python-version` 固定，依赖由 `uv.lock` 固定。当前锁定 PyTorch 2.14.0；Linux 依赖包含 CUDA 13 运行时，服务器驱动必须兼容。正式运行前先检查 `nvidia-smi` 和下面的 CUDA 检查。若服务器驱动不适配，应通过 uv 调整 PyTorch/CUDA 依赖并重新锁定、测试，不使用 pip 手改环境。

```bash
uv run --locked python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

本地已验证 CPU。CUDA bf16/fp16、NCCL 与 8 卡吞吐仍需服务器验收；MPS 训练未开放。DDP 测试使用两个本地 CPU/Gloo 进程，需要允许回环网络通信。

## 五分钟检查完整链路

下面全部是 **synthetic fixture + 微型随机初始化模型**，用于验证程序能运行，不用于报告聊天、推理或 RL 能力。输出目录不可已存在，重复运行需使用新目录。

```bash
uv run --locked moe-lab prepare --input fixtures/pretrain.jsonl --output data/smoke-pretrain --stage pretrain --max-seq-len 64 --val-ratio 0.25
uv run --locked moe-lab train --model-config configs/moe-tiny.json --train-config configs/train-smoke-pretrain.json --data data/smoke-pretrain --output runs/smoke-pretrain

uv run --locked moe-lab prepare --input fixtures/sft.jsonl --output data/smoke-sft --stage sft --max-seq-len 64 --val-ratio 0.25
uv run --locked moe-lab train --model-config configs/moe-tiny.json --train-config configs/train-smoke-sft.json --data data/smoke-sft --init-from runs/smoke-pretrain/step-0000008.pt --output runs/smoke-sft

uv run --locked moe-lab evaluate --checkpoint runs/smoke-sft/step-0000004.pt --data data/smoke-sft
uv run --locked moe-lab generate --checkpoint runs/smoke-sft/step-0000004.pt --chat --prompt '2+1=?' --temperature 0 --max-new-tokens 16
```

几步训练后的输出可能重复或不可读，这是预期行为。要获得有意义的语言能力，需要真实语料和足够训练。

## 正式数据与训练

数据来源、固定版本下载、格式转换和接入命令见 [文本数据指南](docs/data-guide.md)。首轮建议 MiniMind mini 组合（约 2.98 GB 下载）。

预训练 JSONL：

```json
{"text": "一篇完整文档的内容。"}
```

SFT JSONL（可选开头 system，随后 user/assistant 交替，最后为 assistant）：

```json
{"messages": [{"role": "user", "content": "解释什么是专家路由。"}, {"role": "assistant", "content": "专家路由根据输入选择参与计算的专家网络。"}]}
```

固定分词器后整个训练链路保持一致。BPE 训练与 prepare 使用相同的内容哈希切分，**必须保持 seed 和 val-ratio 相同**；tokenizer 命令跳过划为验证集的文档。不要在外部评测集上训练 tokenizer，也不要把同一评测内容混入另一个阶段的数据。

```bash
uv run --locked moe-lab tokenizer --input data/raw/pretrain.jsonl --output data/tokenizer.json --vocab-size 16384 --val-ratio 0.05 --seed 42
uv run --locked moe-lab prepare --input data/raw/pretrain.jsonl --output data/pretrain-v1 --tokenizer data/tokenizer.json --stage pretrain --max-seq-len 512 --val-ratio 0.05 --seed 42
uv run --locked moe-lab prepare --input data/raw/sft.jsonl --output data/sft-v1 --tokenizer data/tokenizer.json --stage sft --max-seq-len 512 --val-ratio 0.05 --seed 42
```

小语料不一定学满 16K 词表。查看 tokenizer 命令输出的实际 `vocab_size`，必要时将模型配置中的值改为一致；训练入口会拒绝不一致的词表。`byte` tokenizer 仅用于快速测试。

先单卡短跑，保留原来的 LR schedule，并在第 10 步保存退出：

```bash
uv run --locked moe-lab train --model-config configs/moe-base.json --train-config configs/train-pretrain.json --data data/pretrain-v1 --tokenizer data/tokenizer.json --output runs/base-pilot --stop-after 10
```

单卡验证后，8 卡独立启动一个正式 run：

```bash
uv run --locked torchrun --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train --model-config configs/moe-base.json --train-config configs/train-pretrain.json --data data/pretrain-v1 --tokenizer data/tokenizer.json --output runs/base-pretrain-001
```

8 卡 SFT：

```bash
uv run --locked torchrun --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train --model-config configs/moe-base.json --train-config configs/train-sft.json --data data/sft-v1 --tokenizer data/tokenizer.json --init-from runs/base-pretrain-001/step-0001000.pt --output runs/base-sft-001
```

`train-pretrain.json` 的 1000 steps 是启动配置，不是“完成预训练”的质量保证。按有效 token 数、验证 loss 与抽样结果决定训练预算。当前默认全局每步最多 8 卡 × 1 样本 × 8 次累积 × 512 tokens；padding 与 SFT mask 会减少有效监督 tokens。

## 恢复与输出

`--init-from` 只加载权重，开启新的优化器与 schedule，用于预训练 → SFT。`--resume` 恢复同一个训练过程，要求模型/训练配置、数据、tokenizer、代码指纹、PyTorch 版本、设备类型和 world size 不变；**恢复也写入新的目录**。

```bash
uv run --locked moe-lab train --model-config configs/moe-base.json --train-config configs/train-pretrain.json --data data/pretrain-v1 --tokenizer data/tokenizer.json --resume runs/base-pilot/step-0000010.pt --output runs/base-pilot-resumed
```

不能将单卡 checkpoint 通过 `--resume` 变成 8 卡 run；可以使用 `--init-from` 开启新实验，但会重置优化器与数据进度。GPU 数值误差可能影响重复运行，CPU 上已验证同配置恢复的逐参数一致性。

每个 run 保存：

- `run.json`：完整配置、参数量、数据/tokenizer/代码指纹、父 checkpoint 指纹、版本与设备信息。
- `metrics.jsonl`：CE、验证 loss/perplexity、辅助损失、路由熵、各层专家占用、学习率、梯度范数、有效 tokens、运行时间与 CUDA 峰值显存。
- `step-*.pt`：模型、优化器、AMP scaler、每 rank 的随机状态、数据 epoch/cursor。checkpoint 较大，主模型的 Adam 状态和权重会占用数 GB，应按磁盘预算调整 save_every。
- `summary.json`：末次指标；BPE run 同时保存 tokenizer 副本。

checkpoint 仅加载张量与基本容器（`weights_only=True`）。不要使用来源不可信的模型文件。

## 阅读顺序与后续计划

1. `src/moe_llm/model.py`：从注意力到专家路由，再到整个解码器。
2. `src/moe_llm/tokenizer.py`、`data.py`：角色标记、监督 mask、shift、切分与 mmap 数据。
3. `src/moe_llm/training.py`：loss 归一化、梯度累积、DDP、评测与恢复。
4. `tests/`：对照测试解释每个模块应该满足的性质。

## 视觉扩展（文本 SFT 完成后使用）

```bash
uv sync --locked --extra vision
uv run --locked --extra vision pytest -q
uv run --locked --extra vision python scripts/smoke.py --output runs/full-smoke-001 --vision
```

视觉训练分 `align`（仅 projector）和 `sft`（projector + Q/V LoRA）。原文本权重和 tokenizer 不变，无图片时视觉 LoRA 关闭，图片会话启用。第一版不更新共享语言权重，不实现始终启用 LoRA 的混合模态训练。

`vision-siglip.json` 是正式视觉配置；`vision-fixture.json` 只用于随机视觉编码器的工程测试。支持一张图、多轮对话、固定图像尺寸、DDP、AMP、梯度累积、恢复、独立 adapter checkpoint。视觉 checkpoint 必须与原文本 SFT 基座和冻结编码器一起使用。

新命令：`vision-prepare`、`vision-train`、`vision-generate`、`vision-evaluate`、`vision-fixture`。`vision-evaluate --text-data ...` 比较 VLM 无图路径与原基座的 logits；`--zero-images` 提供图像置零对照。详见训练指南第 9–13 步。

`train` 和 `vision-train` 均支持 `--eval-max-batches N`：每次定期/末尾验证每卡最多 N 批，不传则全量验证。短跑建议 `--stop-after 9 --eval-max-batches 9`，正式训练示例采用每卡 128 批，并在训练后单独全量评测。独立 `vision-evaluate` 不继承训练上限；SigLIP 对比训练使用配置中的 `eval_samples`。

## 参考项目

本项目的学习路线与实现设计参考了 [MiniMind](https://github.com/jingyaogong/minimind) 和 [MiniMind-V](https://github.com/jingyaogong/minimind-v)：前者提供小型语言模型从网络搭建到预训练、SFT 的实践参考，后者提供视觉编码器、投影层与语言模型连接的 VLM 实践参考。感谢两个项目的开源工作。相关数据来源与许可证另见 [数据指南](docs/data-guide.md) 和训练指南第 10 步。

本项目实现了原生 PyTorch MoE、两种视觉权重来源，以及有图启用、无图关闭的视觉 LoRA，并检查纯文本路径与原始基座一致。MiniMind-V 本身也支持纯文本和 MoE；这些支持不能作为本项目独有的功能，也尚无实验结论证明本项目效果优于参考项目。

[逐步训练指南](docs/training-guide.md) · [实现细节与限制](docs/design.md) · [后续路线](docs/roadmap.md) · [本地验证记录](docs/validation.md)

## 训练终端进度条

文本 `train`、视觉 `vision-train` 和 SigLIP `train` 在交互终端自动显示底部进度条；八卡只由 rank 0 输出。使用 `--epochs N` 时显示当前轮次和本轮优化器步数，例如：

```text
Epoch 1/2 [###---------] 250/1000 train=3.2100 val=3.4800@200 train ETA=900s
```

每次更新的 JSON 指标打印在进度条上方，`metrics.jsonl` 格式保持不变。`train` 是当前更新的训练 loss；`val` 是最近一次实际验证的 loss，`@200` 表示来自全局第 200 步，首次验证前为 `--`。验证仍只在 `eval_every` 和本次结束时执行，不会为了刷新进度条每步验证。验证/保存期间状态为 `validate` / `save`。

轮末显示本轮 100%；下一轮首步切换轮次。恢复从 checkpoint 已完成的步数继续；`--stop-after` 不缩短进度条的完整预算，提前结束显示 `stopped`。未设置 epochs 时显示全局 Steps。ETA 根据本次运行平均更新速度估算，包含已发生的验证/保存等开销，不代表精确完成时间。

重定向到文件或普通管道时自动使用纯 JSON 行，避免控制字符污染日志。`MOE_LAB_PROGRESS=0` 可手动关闭进度条；支持终端控制字符的环境可用 `MOE_LAB_PROGRESS=1` 强制开启（经 `tee` 时文件也会记录控制字符）。无需新增 CLI 参数。

本次代码更新会改变严格恢复的代码指纹。正在运行或需要精确 resume 的旧实验继续使用原代码；完成后再更新，新实验使用新显示。不要为进度条绕过恢复校验。
