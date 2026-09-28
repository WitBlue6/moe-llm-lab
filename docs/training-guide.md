# 从文本 LLM 到 VLM：逐步训练指南

适用目标：完整掌握流程；硬件为一台 8 卡 RTX 3090 服务器。主模型约 264M 总参数、151M 每 token 激活参数（16K 词表时）。

**现在先执行第 0–8 步，训练并评估文本 SFT 基座；通过能力验收后才开始第 9 步视觉训练。** 不需要为了视觉重新预训练语言模型，也不用现在下载图片数据或视觉权重。

目前的测试验证了代码、梯度、数据掩码、恢复和文本路径一致性。你的服务器已运行单卡真实数据的前 9 步，loss 与梯度有限；完整短跑的保存/恢复和八卡 NCCL 仍需验收。下面的学习率、batch 和训练步数均是起点，不能当作已验证的最优参数。

## 0. 把项目放到服务器，建立独立环境

服务器目录按你当前使用的 `~/moe-llm-lab`，不同则修改 `cd` 路径。复制项目时包含 `src/`、`configs/`、`scripts/`、`tests/`、`fixtures/`、`docs/`、`pyproject.toml`、`uv.lock`、`.python-version`。**不要复制 Mac 的 `.venv/` 到 Linux。** 大数据、模型和 runs/ 按需要单独传输。

进入项目根目录，在整个实验期间都从该目录执行命令：

```bash
cd ~/moe-llm-lab
uv --version
uv sync --locked
mkdir -p reports data/raw models runs
uv run --locked moe-lab doctor --output reports/doctor-initial.json
nvidia-smi
nvidia-smi topo -m
```

`doctor-initial.json` 不可已经存在；再次诊断请换文件名。应看到：

- Python 与 `.python-version` 一致；当前为 3.10.17。
- `cuda_available: true`、8 张 GPU、每卡实际显存和 bf16 支持情况。
- 确认卡没有被其他训练占满，检查磁盘空间，给数据与数 GB/个的文本训练 checkpoint 预留空间。

当前锁定 PyTorch 2.14.0；Linux 默认锁文件包含 CUDA 13 运行时依赖。**服务器驱动必须兼容安装的 PyTorch CUDA 运行时**；不要把 `nvcc --version` 或 `nvidia-smi` 上显示的 CUDA 字样当作 PyTorch 已经可用的证明。

如果 CUDA 不可用：先解决驱动与 wheel 兼容问题，可升级驱动，或通过 uv 调整服务器可用的 PyTorch/CUDA 依赖并重新生成锁文件、复测。不要使用 pip 手动修改 `.venv`，也不要在 CPU 上误启动主模型正式训练。先不要进入后续大训练。

若 `uv` 本身尚未安装，先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装；项目不依赖 conda。

## 1. 先做程序验收，不下载正式语料

```bash
uv run --locked pytest -q tests/test_model.py tests/test_data.py tests/test_training.py tests/test_ddp.py
uv run --locked python scripts/smoke.py --output runs/text-smoke-001
uv run --locked moe-lab inspect --model-config configs/moe-base.json
```

smoke 自动执行微型模型的预训练、SFT 与生成，把每个步骤的日志和 checkpoint 放到 `runs/text-smoke-001/`。目录必须不存在；重试换成 `text-smoke-002` 等。

通过条件：测试无失败，smoke 最后显示 `All smoke checks passed`。合成样例训练几步后的生成可能重复、不可读，这是正常的；**它不证明模型学会了聊天。** `moe-tiny.json` 与 `byte` tokenizer 仅用于工程测试。

DDP 测试使用两个 CPU/Gloo 进程，需要本机回环网络。它证明分布式梯度逻辑，并不能代替后面真实 GPU 的短跑。

## 2. 准备两份正式文本数据

已提供可直接执行的 [数据选择、下载与转换指南](data-guide.md)：推荐先用 MiniMind mini 组合。你已将转换结果放到 `data/raw/pretrain.jsonl`、`data/raw/sft.jsonl`，与本篇命令一致；保留 `data/raw/import-report.json` 和下载目录中的来源记录，不需要重新转换。

仓库没有附带正式训练语料。你需要选择许可明确、质量合适的数据，转换成下面两个 UTF-8 JSONL 文件；每行一个 JSON 对象，不能把整个文件写成一个 JSON 数组。

### 2.1 预训练数据：`data/raw/pretrain.jsonl`

```json
{"text":"一篇完整的中文或英文文档。"}
{"text":"另一篇独立文档，保留合理标点与段落。"}
```

作用是学习语言、知识与续写能力。优先清理乱码、广告、无意义重复、泄漏的测试题及质量低下的机器生成文本。长文档会在预处理时分窗，不要求你提前截到 512 tokens。

### 2.2 指令数据：`data/raw/sft.jsonl`

```json
{"messages":[{"role":"user","content":"解释什么是梯度下降。"},{"role":"assistant","content":"梯度下降通过沿损失函数梯度的反方向更新参数，逐步降低损失。"}]}
{"messages":[{"role":"system","content":"请使用简洁的中文回答。"},{"role":"user","content":"什么是注意力机制？"},{"role":"assistant","content":"注意力机制根据当前输入，为其他位置的信息分配不同权重。"},{"role":"user","content":"它有什么作用？"},{"role":"assistant","content":"它让模型在当前预测中更充分地利用相关上下文。"}]}
```

支持可选的开头 system，以及交替的 user/assistant，最后必须为 assistant。当前不支持 tool 角色；不要把带工具调用的原始数据直接塞入。不要在普通文本里写本项目保留的 `<|bos|>`、`<|eos|>`、`<|user|>` 等控制字符串。

原始数据若使用 `instruction/input/output` 或其他字段，需要先转换成上述格式。不能把普通文章简单套成 assistant 内容并期待它自动变成好的问答数据。

另外单独保存一份**不参与任何训练**的固定评测题，至少覆盖中文问答、指令遵循、简单推理和多轮对话。不要只观察训练样本的生成。

建议用 `data/raw/SOURCES.md` 记录每份数据的来源、版本/下载日期、许可证、筛选方法与用途。CLI 会记录文件 SHA256，但不会自动判断许可证、内容质量或跨数据集泄漏。

## 3. 训练并固定 tokenizer

在预训练语料上训练 16K ByteLevel BPE：

```bash
uv run --locked moe-lab tokenizer \
  --input data/raw/pretrain.jsonl \
  --output data/tokenizer-v1.json \
  --vocab-size 16384 --val-ratio 0.05 --seed 42
```

`tokenizer` 会按与 `prepare` 相同的内容哈希规则跳过验证文档。后面的预训练数据准备必须使用相同的 `val-ratio` 和 `seed`。不要另行把验证集加进 tokenizer 语料。

产物：`data/tokenizer-v1.json` 和 `.meta.json`。检查实际 `vocab_size`、训练文档数、留出文档数。小语料可能学不满 16K；用以下命令生成与实际词表一致的新配置：

```bash
uv run --locked moe-lab configure \
  --model-config configs/moe-base.json \
  --tokenizer data/tokenizer-v1.json \
  --output configs/model-text-v1.json
uv run --locked moe-lab inspect --model-config configs/model-text-v1.json
```

开始正式预训练后，**不要重新训练或修改这个 tokenizer**。预训练、SFT、视觉对齐、视觉 SFT 及推理都使用同一个 tokenizer。视觉接入不需要新增 token ID。

## 4. 预处理文本数据，检查留出集和 token 数

先使用 512 tokens 的训练窗口，降低初期显存压力：

```bash
uv run --locked moe-lab prepare \
  --input data/raw/pretrain.jsonl --output data/pretrain-v1 \
  --tokenizer data/tokenizer-v1.json --stage pretrain \
  --max-seq-len 512 --val-ratio 0.05 --seed 42

uv run --locked moe-lab prepare \
  --input data/raw/sft.jsonl --output data/sft-v1 \
  --tokenizer data/tokenizer-v1.json --stage sft \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
```

产物包括 `manifest.json`、训练/验证二进制数据与索引。查看：

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
for name in ('pretrain-v1', 'sft-v1'):
    m = json.loads((Path('data') / name / 'manifest.json').read_text())
    print(name)
    print('records:', m['split_records'])
    print('tokens:', m['tokens'])
    print('filtering:', m['stats'])
PY
```

通过条件：训练和验证均非空；预训练数据能形成合理数量的 tokens；SFT 的 `overlong_sft` 没有意外丢掉大多数样本。

当前预处理规则：

- exact dedup 后按文档内容哈希切分，再对预训练长文档分窗，避免同一文档窗口进入两边。
- SFT 只对 assistant 回答与其 EOS 计算损失。
- SFT 对话加角色标记后超出 512 tokens，会整条跳过，不偷偷截断回答。
- 若过长样本很多，可以清理/缩短对话，或用新的输出目录重新准备 1024-token 数据；模型配置上限 2048 不是“已学会 2048 上下文”的保证。
- exact dedup 不负责近重复、跨语料泄漏或同题不同答案分组；正式评测需要额外检查。

不要改动已被某个 run 使用的预处理文件。源数据变化时创建 `pretrain-v2` 等新版本。

## 5. 确认训练预算，做单卡和八卡短跑

首次使用时复制训练配置，之后所有正式命令都用这份固定文件。你已创建下面两份配置，继续第 5.1 步即可，不要重复执行创建命令或覆盖配置：

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
for source, target in [('train-pretrain.json', 'text-pretrain-v1.json'),
                       ('train-sft.json', 'text-sft-v1.json')]:
    config = json.loads((Path('configs') / source).read_text())
    with (Path('configs') / target).open('x') as f:
        json.dump(config, f, indent=2)
PY
```

初始预训练配置：单卡 microbatch=1，梯度累积=8，bf16，activation checkpointing 已在模型配置启用。八卡、512-token 窗口时，每个 optimizer step 的输入 token 上限为：

```text
8 GPUs × 1 sample/GPU × 8 accumulation × 512 tokens = 32,768 tokens
```

padding、短文档及 SFT 的回答 mask 会使有效监督数更少；实际看日志的 `supervised_tokens_this_step` 和 `trained_tokens`。

模板的 `max_steps=1000` 只是起点，八卡上限约 3277 万输入 tokens，不代表“基础模型已经训练充分”。建议先根据真实数据做学习曲线，再考虑扩大到更高 token 预算。修改总步数/学习率/schedule 必须在正式 run 之前完成；精确恢复不允许中途换 schedule。

### 5.1 单卡检查显存与 loss

先同步本地更新的 `src/moe_llm/cli.py`、`src/moe_llm/training.py` 和本指南到服务器。在服务器确认帮助里有 `--eval-max-batches`：

```bash
uv run --locked moe-lab train --help
```

若旧版训练还在全量验证，先在其终端按 Ctrl+C 停止，等进程退出再运行下面命令。旧进程不会自动采用新代码；未完成验证时可能还没有 checkpoint。新目录也已存在时，将后缀换成未使用的名称。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-pretrain-v1.json \
  --data data/pretrain-v1 --tokenizer data/tokenizer-v1.json \
  --output runs/text-pretrain-pilot-1gpu-v2 --stop-after 9 --eval-max-batches 9
```

`--stop-after 9` 只提前结束，不修改原计划的 LR schedule。`--eval-max-batches 9` 将每次验证限制为每卡最多 9 个批次，单卡 batch_size=1 时最多 9 条验证样本；不使用训练样本替代验证集。检查日志无 NaN，成功产生 `step-0000009.pt` 和 `summary.json`。这是运行检查，少量样本的 loss 不代表全量验证结果。

更新服务器上的 `src/moe_llm/cli.py` 和 `src/moe_llm/training.py` 后，新参数才可使用。已有进程不会自动采用新代码。输出目录须为新目录。不传 `--eval-max-batches` 时仍按原规则进行全量验证；日志用 `val_full`、`val_records` 与 `val_total_records` 区分验证范围。

### 5.2 八卡检查 NCCL、吞吐与显存

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-pretrain-v1.json \
  --data data/pretrain-v1 --tokenizer data/tokenizer-v1.json \
  --output runs/text-pretrain-pilot-8gpu-v2 --stop-after 10 --eval-max-batches 9
```

八卡当前 batch_size=1，每卡最多 9 个验证批次，总计最多 72 条样本；与单卡短跑的 9 条不是同一验证范围，不要直接比较二者 val_loss。结束应产生 `runs/text-pretrain-pilot-8gpu-v2/step-0000010.pt` 和 `summary.json`。

所有 GPU 都应有进程和活动，训练、验证、保存均应正常。观察 `summary.json` 的峰值显存、时间及有效 tokens/s；十步短跑包含启动等开销，只适合初步判断，可用更长的独立短跑估计稳定吞吐。

粗略时间估计为 `目标有效 tokens / 实测有效 tokens/s`，另留 checkpoint、验证、数据处理开销。不要因为有八卡就假定必然八倍加速；初版 MoE 逐专家调度和每微批 DDP 同步以可读、正确为优先。

若显存不足：保持 batch=1，先把新数据窗口降到 256，再检查其他进程占用和 checkpointing；不要直接关闭 MoE 专家或改模型形状继续加载旧权重。每卡显存独立，DDP 不会把 8 张卡合成一个 192GB 显存池。

## 6. 正式文本预训练

短跑通过后，用确定好的配置开启一个新的 run：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-pretrain-v1.json \
  --data data/pretrain-v1 --tokenizer data/tokenizer-v1.json \
  --output runs/text-pretrain-001 --eval-max-batches 128
```

这里每次定期验证和结束验证均限制为每卡 128 批；八卡 batch_size=1 时是验证集固定前 1024 条（不足则全部），用于观察学习曲线。保持相同 GPU 数、batch 和上限，才能比较同一验证范围；这不是随机抽样，也不保证代表完整验证集。它不限制训练数据量。仍按配置的 `eval_every` 触发，默认每 100 步一次，结束时也验证；此例结束时不会自动做全量验证。

在持久终端会话中运行，避免关闭 SSH 后中断。不要让另一个进程写同一个输出目录。

每次检查：

- `train_loss`、`val_loss` 是否有合理趋势；不只看训练 loss。
- `expert_usage` 是否长期塌缩到少数专家，`router_entropy` 和 `aux_loss` 是否异常。
- 梯度范数是否有限，吞吐/显存是否异常波动。
- `trained_tokens` 与预期预算是否一致；反复重放小语料会增加这个计数，但不会增加独立数据量。

用保存下来的 checkpoint 做普通续写，**预训练阶段不要加 `--chat`**：

```bash
uv run --locked moe-lab generate \
  --checkpoint runs/text-pretrain-001/step-0001000.pt \
  --tokenizer data/tokenizer-v1.json \
  --prompt '机器学习是一种' --max-new-tokens 128 --temperature 0.8
```

这里的 `step-0001000.pt` 假设你保持模板总步数 1000；若修改了步数，请使用实际存在的 checkpoint。不要只选最后一个：结合验证 loss 与固定续写样例选择。若仍然乱码、严重重复或缺乏基本连贯性，先排查数据/tokenizer/训练预算，不能指望 SFT 或 RL 修复所有问题。

### 预训练结束后做一次独立全量验证

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab evaluate \
  --checkpoint runs/text-pretrain-001/step-0001000.pt \
  --data data/pretrain-v1 --tokenizer data/tokenizer-v1.json \
  --device cuda --batch-size 1
```

这是单卡全量验证，会遍历当前 68,104 个验证样本，耗时可能很长；不要把它当作短跑。独立 `evaluate` 使用 fp32，而训练内验证使用配置的 bf16，比较数值时同时记录精度和验证范围。当前 `evaluate` 不支持 `--eval-max-batches`，也不会自动保存报告；保留终端输出。可在另一个终端使用 `nvidia-smi` 观察活动。

### 中断后恢复

保持同一代码、环境、模型/训练配置、数据、tokenizer、设备类型与 GPU 数，用新的输出目录恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-pretrain-v1.json \
  --data data/pretrain-v1 --tokenizer data/tokenizer-v1.json \
  --resume runs/text-pretrain-001/step-0000500.pt \
  --output runs/text-pretrain-001-resumed --eval-max-batches 128
```

示例要求第 500 步文件实际存在，且计划总步数大于 500。恢复示例也保留同一验证上限。如果从恢复目录得到最终 checkpoint，后续 generate、evaluate 和 SFT 的 `--init-from` 都要改成恢复目录下实际选中的文件；不要继续引用旧目录里不存在的最终文件。`--resume` 恢复优化器、随机数和数据位置；`--init-from` 仅继承模型权重并开启新训练。不要互换。换代码后严格 resume 会拒绝恢复，所以正式训练开始后先固定项目版本。

## 7. 全参数文本 SFT

确认预训练基座具备基本语言能力后，使用前面准备的 `sft-v1`，先从预训练 checkpoint 做单卡短跑：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-sft-v1.json \
  --data data/sft-v1 --tokenizer data/tokenizer-v1.json \
  --init-from runs/text-pretrain-001/step-0001000.pt \
  --output runs/text-sft-pilot-1gpu-v1 \
  --stop-after 9 --eval-max-batches 9
```

应保存 `runs/text-sft-pilot-1gpu-v1/step-0000009.pt` 和 `summary.json`。短跑与正式 SFT 均从选中的预训练权重开始，正式运行不接续短跑。

短跑通过后，启动八卡正式 SFT：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli train \
  --model-config configs/model-text-v1.json \
  --train-config configs/text-sft-v1.json \
  --data data/sft-v1 --tokenizer data/tokenizer-v1.json \
  --init-from runs/text-pretrain-001/step-0001000.pt \
  --output runs/text-sft-001 --eval-max-batches 128
```

正式 SFT 每次验证最多每卡 128 批，默认总步数 500、较小学习率只是起始配置；同样用验证趋势及固定问答判断预算。SFT checkpoint 必须仍使用原 tokenizer 和同一网络形状。

SFT 的恢复命令与预训练相同，换成 SFT 配置、数据和 `--resume` 的 SFT checkpoint，移除 `--init-from`，保留 `--eval-max-batches 128`，且写入新的输出目录。

## 8. 验收并保存纯文本基线

下面第一条是训练后单独执行的全量 SFT 验证：当前共 36,571 条，单卡 fp32、batch_size=1，可能需要较长时间；它不会因训练命令使用过验证上限而变成子集验证。第二条才是生成问答。文件名假设 SFT 总步数为 500；若改过总步数或从恢复目录完成训练，统一替换这里及后续视觉命令的基座路径。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab evaluate \
  --checkpoint runs/text-sft-001/step-0000500.pt \
  --data data/sft-v1 --tokenizer data/tokenizer-v1.json \
  --device cuda --batch-size 1

uv run --locked moe-lab generate \
  --checkpoint runs/text-sft-001/step-0000500.pt \
  --tokenizer data/tokenizer-v1.json --chat \
  --prompt '请用三句话解释为什么需要训练集和验证集。' \
  --max-new-tokens 160 --temperature 0
```

通过条件：能理解基本指令、输出相对连贯的回答；在不参与训练的固定问题上有可接受表现；不存在严重重复、乱码、角色串线。验证 CE/perplexity 只是一部分，不能代替答案质量检查。

记录并固定选中的 SFT checkpoint 路径、它的 tokenizer、配置、固定评测结果。后续视觉 checkpoint 会绑定其 SHA256。不要移动后又丢失它，也不要用新的训练覆盖它。

**到这里，你要先开展的基础文本训练就完整了。** DPO、PPO 尚未实现；不需要等它们才能开始下一阶段视觉扩展。

## 9. 视觉网络与权重来源：二选一

**两条路线都使用项目内手写的 PyTorch 视觉网络**（`src/moe_llm/siglip.py`）：patch embedding、位置编码、多头注意力、MLP、LayerNorm。运行时不调用 Transformers 的 `SiglipVisionModel.from_pretrained`；Transformers 暂保留在可选依赖中，用于数值对照测试。

| 路线 | 网络实现 | 权重来源 | 适用目的 |
| --- | --- | --- | --- |
| A：加载已有权重 | 本项目手写网络 | Google 预训练 SigLIP 参数 | 先验证 VLM 连接层与视觉 LoRA |
| B：自己训练 | 本项目手写网络＋独立文本编码器 | 两个编码器随机初始化，用图文对训练 | 学习视觉编码器预训练全过程 |

路线 B 是小规模 **SigLIP 式训练**：采用 sigmoid 图文匹配损失、图像 mean pooling、文本 EOS pooling 和投影头；并非官方完整架构/训练配方复现，不保证达到官方预训练效果。现有 LLM 的 tokenizer 只负责切分描述文字，文本对比编码器也是新建并随机初始化，不加载或更新你的 LLM。训练完成后只导出视觉 backbone，文本对比编码器和对比投影头不会接入 VLM。

安装依赖并做工程检查：

```bash
uv sync --locked --extra vision
uv run --locked --extra vision pytest -q
uv run --locked --extra vision python scripts/smoke.py --output runs/full-smoke-001 --vision
```

该 smoke 使用随机冻结的 fixture 编码器，不证明识图能力。新增的 SigLIP 测试覆盖原生/参考 patch 特征与预处理一致性、损失数学、跨卡负样本梯度、保存恢复、导出加载和无图文本保留。尚未在你的服务器验证新编码器的八卡吞吐或真实训练效果。

此次需要同步 `src/`、`scripts/`、`configs/`、`tests/`、`docs/`、`pyproject.toml` 和 `uv.lock`。代码指纹会变化；已经开始的旧 run 若需要严格恢复，应保留其旧版代码/环境，不能用新版强行 resume。已完成的文本 SFT checkpoint 可以作为新版视觉训练的 `--base-checkpoint`。

### 9A. 加载兼容预训练权重

```bash
uv run --locked --extra vision python scripts/download_siglip_weights.py \
  --output models/siglip-base-patch16-224
```

仅下载固定版本 `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` 的配置、预处理配置和 safetensors，不下载执行模型仓库代码。权重文件约 813 MB（原文件还含不使用的文本参数）；加载器只读取视觉 patch 路径所需张量，严格校验键名和形状。下载验证权重 SHA256，保存来源记录。此路线是“手写网络＋外部预训练权重”，不能称为权重也从零训练。

如果之前已经下载过该目录，可保留原始配置/权重直接使用；新的下载脚本对没有自身下载记录的非空目录会拒绝写入，避免覆盖。网络不可达时可尝试给下载命令加 `HF_ENDPOINT=https://hf-mirror.com`，镜像可用性由第三方决定，校验仍必须通过。

选择配置（新文件，不覆盖）：

```bash
uv run --locked python - <<'PY'
from pathlib import Path
with Path('configs/vision-active.json').open('x') as f:
    f.write(Path('configs/vision-siglip.json').read_text())
PY
```

原生加载器当前兼容第一代 SigLIP 固定分辨率视觉 backbone；不能将 SigLIP2/NaFlex 仓库直接替换进去。默认 224×224、patch16，14×14 patch 经平均池化到 8×8，共 64 个视觉 tokens。

然后执行第 10.1–10.2 步准备视觉数据，跳过第 10.3 步的从零训练，进入第 11 步。

### 9B. 自己训练视觉权重

**跳过 9A 的权重下载和配置选择。** 先执行第 10 步的数据下载、转换和图文对预训练；第 10.3 步训练验收并导出后再选择 `vision-active.json`。

## 10. 下载视觉数据，路线 B 再进行图文对预训练

### 10.1 数据来源与下载

当前推荐 [MiniMind-V 数据仓库](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/1e279a8b665cb10383451a6af6fd62b9f35bdd79)，固定版本 `1e279a8b665cb10383451a6af6fd62b9f35bdd79`，核对日期 2026-09-28。

| 文件 | 下载大小（十进制） | 本项目用途 |
| --- | ---: | --- |
| `pretrain_i2t.parquet` | 4.33 GB | 路线 B 的图片—描述对比预训练；两条路线的 projector 对齐 |
| `sft_i2t.parquet` | 4.93 GB | 图文指令 SFT，排除纯文本占位样本 |

数据卡说明 caption 约 127 万条、约 64 万张唯一图片，SFT 约 290 万条，后者混有 caption 与纯文本数据。我们会按图片字节去重图像文件，并且对比预训练仅保留每张图片首次遇到的有效描述，避免同一图片的多条描述在 batch 中被错误当作负例；不保证中英文均衡。转换后的确切数量看报告，不能直接沿用数据卡数字。

图片以 `image_bytes` 内嵌在 Parquet 中，问答在 `conversations` 列，**无需另找图片压缩包**。网页自动 Viewer 可能只展示几张测试图，不代表训练文件只有几条。上游主要来自 [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)，包含生成式标注；该上游数据卡标注 CC-BY-NC-4.0（非商业），不能因 MiniMind-V 聚合仓库标注 Apache-2.0 就当成全部可商用。还需核对原始图片和其他混入来源条款。

```bash
uv run --locked --extra vision python scripts/download_visual_data.py \
  --stage both --output data/downloads/minimind-vision-v1 --list

uv run --locked --extra vision python scripts/download_visual_data.py \
  --stage both --output data/downloads/minimind-vision-v1
```

合计约 9.26 GB，仅下载这两份文件，固定版本并校验大小和 SHA256。中断后重跑同一命令。下载计划不同则使用新目录。还需给解包图片、JSONL、索引和训练 checkpoint 留空间，建议先预留至少 50 GB 数据空间，checkpoint 另计；几十万图片也需要足够 inode。脚本已用合成 Parquet 测试，但未在本机完整下载和清洗这两份真实文件。

### 10.2 转换格式并提取图片

```bash
uv run --locked --extra vision python scripts/convert_visual_data.py \
  --pretrain data/downloads/minimind-vision-v1/pretrain_i2t.parquet \
  --sft data/downloads/minimind-vision-v1/sft_i2t.parquet \
  --output data/raw/vision-v1
```

输出目录必须尚不存在，得到：

- `vision-align.jsonl`：图片描述会话，用于投影层对齐。
- `vision-sft.jsonl`：图片指令会话，用于视觉 SFT。
- `captions.jsonl`：每张图片一份描述，用于路线 B。
- `images/`：以内容 SHA256 命名的原始图片文件；扩展名 `.image`，Pillow 根据实际文件内容解码。
- `import-report.json`：来源/输出哈希、保留数、过滤原因、转换器版本指纹。

脚本流式读取 Parquet、保留原始图片字节，移除首个 user 问题中的 `<image>` 标记。过滤多图、工具调用、无图、8×8 等极小占位图、不合法对话，按普通回答策略处理 reasoning 字段。先用 `--limit 1000` 和新的 pilot 目录可以检查转换流程，但它只是文件前缀，不是代表性训练样本；正式训练不要误用 pilot。

本项目支持一段会话一张图片。转换后的格式类似：

```json
{"image":"ab/完整哈希.image","messages":[{"role":"user","content":"描述这张图片。"},{"role":"assistant","content":"一只猫坐在窗边。"}]}
```

不要在转换后的文本中手动添加图像特殊 token；VLM 预处理会自动插入 64 个视觉位置，并屏蔽这些位置的语言监督。

### 10.3 仅路线 B：从随机初始化训练 SigLIP 式编码器

**路线 A 跳过这一节。** 使用 `pretrain_i2t.parquet` 提取的 caption 学图片—描述匹配，不使用任意 VQA 答案替代完整图片描述。

先准备对比数据，沿用已有 tokenizer；不重新训练或修改它：

```bash
uv run --locked --extra vision python -m moe_llm.siglip_training prepare \
  --input data/raw/vision-v1/captions.jsonl \
  --image-root data/raw/vision-v1/images --output data/siglip-pairs-v1 \
  --tokenizer data/tokenizer-v1.json --text-length 128 --val-ratio 0.05 --seed 42
```

图片按内容哈希划分 train/val，规则与后面的 VLM 划分相同；必须保持 seed=42、val_ratio=0.05，并保留图片原始字节。描述在 128-token 上限内保留 BOS/EOS，超长部分截断，统计在 manifest；这与 VLM 超长整段跳过的策略不同。近重复图、重编码图还需额外清理。

模型配置 `configs/siglip-scratch.json`：视觉 6 层、hidden384、6 头、patch16；文本对比编码器 4 层；对比 embedding256。它是独立于 264M LLM 的小模型。

单卡短跑（默认每卡 batch16）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision python -m moe_llm.siglip_training train \
  --config configs/siglip-scratch.json --data data/siglip-pairs-v1 \
  --tokenizer data/tokenizer-v1.json --output runs/siglip-pilot-1gpu --stop-after 10
```

八卡短跑：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.siglip_training train \
  --config configs/siglip-scratch.json --data data/siglip-pairs-v1 \
  --tokenizer data/tokenizer-v1.json --output runs/siglip-pilot-8gpu --stop-after 10
```

此训练器与 `moe-lab train` 分开，**不接收 `--eval-max-batches`**。配置的 `eval_samples=256` 限制验证候选集，短跑末尾也最多评估 256 个图文对，打印 `validation_start`，不跑全部验证集。若需减少短跑验证，先复制配置并修改 eval_samples，不改正在运行的文件。

八卡默认全局 batch 为 128。每个图像与所有卡的文本特征做 sigmoid 正/负配对，使用带梯度的 all-gather。当前不实现梯度累积；普通梯度累积不会扩大同一步对比负样本池。单卡和八卡负样本数不同，训练 loss 不可直接横比。数据采样丢弃不足全局 batch 的尾部，保证不同卡形状一致，不通过重复图片补齐。

先检查两侧编码器梯度、loss 有限、保存恢复正常，再启动新目录的正式实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.siglip_training train \
  --config configs/siglip-scratch.json --data data/siglip-pairs-v1 \
  --tokenizer data/tokenizer-v1.json --output runs/siglip-scratch-001
```

默认 max_steps=10000，八卡 batch16 时处理约 128 万图文对（包含跨 epoch 重复）；这是实验起点，不是官方级预训练预算。每 500 步及结束时用固定最多 256 个留出图文对验证，每 1000 步及结束时保存。学习率/总步数需在正式运行前确定。日志记录全局 batch、训练图片数、耗时、分配 GPU 秒、峰值显存；run/checkpoint 保存数据、代码、tokenizer、配置和随机状态，无外部模型 API 调用。

精确恢复示例，要求第 1000 步文件存在，保持代码、配置、环境、数据和 GPU 数不变：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.siglip_training train \
  --config configs/siglip-scratch.json --data data/siglip-pairs-v1 \
  --tokenizer data/tokenizer-v1.json \
  --resume runs/siglip-scratch-001/step-0001000.pt --output runs/siglip-scratch-001-resumed
```

**验收视觉预训练，不只看 loss：** 固定候选集上的 `image_to_text_r1` / `text_to_image_r1` 是图到文/文到图检索 Recall@1。256 个候选随机基线为 1/256；与早期 checkpoint 在同一集合上比较，并人工检查检索结果。没有统一合格阈值，不能把小集合 R@1 当作 VQA 能力。较大的评测候选集计算 N×N 相似度矩阵，不要未经显存评估直接覆盖整个数据集。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra vision python -m moe_llm.siglip_training evaluate \
  --checkpoint runs/siglip-scratch-001/step-0010000.pt \
  --data data/siglip-pairs-v1 --device cuda --batch-size 16 --max-samples 256
```

检查学习趋势后导出选中的视觉 backbone（这里示例用最终步，若从恢复目录完成需修改路径）：

```bash
uv run --locked --extra vision python -m moe_llm.siglip_training export \
  --checkpoint runs/siglip-scratch-001/step-0010000.pt \
  --output models/siglip-scratch-v1

uv run --locked python - <<'PY'
from pathlib import Path
with Path('configs/vision-active.json').open('x') as f:
    f.write(Path('configs/vision-siglip-scratch.json').read_text())
PY
```

导出包含网络配置、预处理配置、safetensors 权重、来源/训练 checkpoint 指纹。导出成功只说明格式可用，不表示效果合格。不要导出十步 pilot 后当作正式视觉模型。路线 B 可能明显弱于路线 A；可以作为对照实验，但不要承诺同样成本得到同等效果。两条路线的视觉 hidden size 不同，必须各自重新训练 projector/视觉 LoRA，不能直接互换适配器 checkpoint。

### 10.4 两条路线汇合：准备 VLM 对齐与 SFT 数据

此时 `configs/vision-active.json` 必须指向你选中的视觉权重目录；路线 B 要先完成导出。只选择一次，开始视觉训练后不要原地修改。切换路线应创建新的配置和新的 runs 目录。

```bash
uv run --locked --extra vision moe-lab vision-prepare \
  --input data/raw/vision-v1/vision-align.jsonl --image-root data/raw/vision-v1/images \
  --output data/vision-align-v1 --tokenizer data/tokenizer-v1.json \
  --vision-config configs/vision-active.json --max-seq-len 512 --val-ratio 0.05 --seed 42

uv run --locked --extra vision moe-lab vision-prepare \
  --input data/raw/vision-v1/vision-sft.jsonl --image-root data/raw/vision-v1/images \
  --output data/vision-sft-v1 --tokenizer data/tokenizer-v1.json \
  --vision-config configs/vision-active.json --max-seq-len 512 --val-ratio 0.05 --seed 42
```

同图所有问题按原始图片内容哈希分到同一侧；保持与对比预训练相同的 seed 和比例。这些文件在不同阶段有图片重叠，但相同图片不会从某阶段训练侧跑到另一个阶段验证侧。这里 512 上限包含 64 个视觉 tokens 和全部对话，超长整段跳过；caption 可能较长，要先看 manifest 的 `overlong` 和剩余样本数。如果大多数被过滤，用新目录准备更长窗口并做显存短跑。

图片不可准备后悄悄替换；搬迁图片目录时通过 `--image-root` 指向相同内容的新位置。文本基座仍必须通过第 8 步验收，才能开始后面的正式 VLM 训练。

## 11. 视觉对齐：只训练投影层

先完成第 9 步的 fixture smoke，再执行正式训练。**当前视觉训练没有 `--eval-max-batches` 参数**；`vision-train --stop-after 10` 仍会在最后一步跑完整视觉验证集，不能套用文本短跑命令来避免长等待。下面的正式视觉训练仍每 100 步及结束时全量验证，验证结束后才打印该步日志并保存 checkpoint，请预留时间：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint runs/text-sft-001/step-0000500.pt \
  --vision-config configs/vision-active.json --train-config configs/train-vision-align.json \
  --data data/vision-align-v1 --tokenizer data/tokenizer-v1.json \
  --output runs/vision-align-001
```

冻结视觉编码器和文本 LLM，只有 projector 进入优化器。LLM 前向仍保留对输入的梯度，使损失能够回传到 projector。图文回答使用与文本 SFT 相同的 assistant-only 监督原则，视觉位置不作为预测目标。

检查 loss/验证趋势、显存和训练参数量。冻结 LLM 不等于训练几乎不占显存，仍需其前向与向输入反传的激活。

## 12. 视觉 SFT：投影层＋视觉 LoRA

同样按视觉训练配置定期及结束时进行全量验证。以下假设视觉对齐训练了 1000 步并在指定目录保存；调整了总步数或恢复目录时，必须同步修改 `--init-from`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked --extra vision torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli vision-train \
  --base-checkpoint runs/text-sft-001/step-0000500.pt \
  --vision-config configs/vision-active.json --train-config configs/train-vision-sft.json \
  --data data/vision-sft-v1 --tokenizer data/tokenizer-v1.json \
  --init-from runs/vision-align-001/step-0001000.pt \
  --output runs/vision-sft-001
```

默认 LoRA 作用在各层注意力的 Q/V 投影，rank=8；MoE router、专家、embedding 和原始注意力权重都保持冻结。投影层和 LoRA 使用不同学习率，记录在 run.json/metrics.jsonl。

这个版本采用**视觉适配器模式**：有图片时启用视觉 LoRA，无图片时关闭。它没有实现“始终开启 LoRA 的图文＋纯文本混合更新”训练器；因为原始文本路径完全不更新，所以本模式无需靠文本回放来抵消遗忘。后续可以把统一混合训练作为独立对照实验。

恢复同一视觉 run 时改用 `--resume`，保留相同 `--base-checkpoint`、配置、数据、GPU 数与代码，并换新输出目录。不要把视觉适配器 checkpoint 当成独立完整 LLM；推理还需要原始 SFT 基座和同一份视觉编码器。

## 13. 视觉能力与文本保留验收

下面会依次全量运行正常图片验证、`--zero-images` 对照，并对整个文本验证集比较两套模型的 logits，耗时可能明显长于单次验证。目前不支持验证批次数上限；不要当作快速 smoke。文件名假设视觉 SFT 完成了 1000 步。

```bash
uv run --locked --extra vision moe-lab vision-evaluate \
  --base-checkpoint runs/text-sft-001/step-0000500.pt \
  --checkpoint runs/vision-sft-001/step-0001000.pt \
  --tokenizer data/tokenizer-v1.json --data data/vision-sft-v1 \
  --text-data data/sft-v1 --zero-images \
  --output reports/vision-sft-001-eval.json
```

检查 `text_logits_identical` 与 `text_max_logit_error`。在相同 fp32 推理路径上，关闭视觉 LoRA 后应与原文本基座一致；本地测试为完全一致。该检查会同时加载基座和 VLM，额外占用一份 LLM 显存，必要时用 `--device cpu`。

`val_loss` 是正常图片条件下的回答 loss；`zero_image_loss` 是将归一化图片张量置零的对照。如果差异很小，值得排查模型是否忽略图片；它本身不是视觉理解成功或失败的充分证明。还应在留出图片上人工检查答案，并将同一问题配不同图片，观察答案是否随图片内容合理变化。

有图推理，从实际准备好的视觉验证集取第一张图（不再使用不存在的示例图片路径）：

```bash
VLM_TEST_IMAGE=$(uv run --locked python - <<'PYIMAGE'
import json
from pathlib import Path
root = Path('data/vision-sft-v1')
manifest = json.loads((root / 'manifest.json').read_text())
with (root / 'val.jsonl').open() as f:
    row = json.loads(next(f))
print(Path(manifest['image_root']) / row['image'])
PYIMAGE
)

uv run --locked --extra vision moe-lab vision-generate \
  --base-checkpoint runs/text-sft-001/step-0000500.pt \
  --checkpoint runs/vision-sft-001/step-0001000.pt \
  --tokenizer data/tokenizer-v1.json --image "$VLM_TEST_IMAGE" \
  --prompt '描述这张图片。' --temperature 0 --max-new-tokens 128
```

无图推理，走原始文本路径：

```bash
uv run --locked --extra vision moe-lab vision-generate \
  --base-checkpoint runs/text-sft-001/step-0000500.pt \
  --checkpoint runs/vision-sft-001/step-0001000.pt \
  --tokenizer data/tokenizer-v1.json \
  --prompt '请解释梯度下降。' --temperature 0 --max-new-tokens 128
```

图像多轮对话可用 `--messages-file data/conversation.json` 替换 `--prompt`。文件是消息数组，末尾为 user，仍要传入同一张 `--image`；不能在后续轮次省略图片后声称模型仍记得它。CLI 每次重新构造整个上下文，内部生成过程只在 prefill 编码一次图片，后续 tokens 复用 KV cache。

## 常见问题定位

| 现象 | 优先检查 |
| --- | --- |
| CUDA unavailable | GPU 驱动与锁定的 PyTorch CUDA wheel；先通过 doctor |
| uv 找不到包/无法下载 | 网络、包镜像和 uv 配置；不要删锁文件绕过复现 |
| 显存不足 | batch、序列长度、其他 GPU 进程、checkpointing；视觉 tokens 也占上下文 |
| 最后一步或每 100 步长期没有新日志 | 文本训练检查 `validation_start` 的样本数；短跑加 `--eval-max-batches 9`，正式文本训练用固定子集并另做全量评测；视觉暂为全量验证 |
| unrecognized arguments: --eval-max-batches | 同步更新文本 cli.py 和 training.py；该参数仅用于 `train`，不支持 `evaluate` 或 `vision-train` |
| 配置/词表不匹配 | tokenizer 是否更换，是否使用 configure 生成的模型配置 |
| SFT 样本几乎全跳过 | manifest 的 overlong_sft；整理数据或用新目录准备更长窗口 |
| 输出目录已存在 | 更换新目录；恢复也用新目录，不覆盖已有实验 |
| resume 被拒绝 | 是否改了代码、版本、schedule、数据、tokenizer 或 GPU 数 |
| 视觉 checkpoint 无法加载 | 是否提供同一个文本 SFT 基座、同样的编码器文件和结构配置 |
| torchrun 本机主机名解析失败 | 用下面的显式回环 rendezvous 参数，仅适用于单机 |
| 视觉 loss 降但回答不看图 | 检查图像/问答配对、视觉插入位置、遮图与换图对照、数据质量 |

单机主机名有问题时，可将 `--standalone` 替换为：

```text
--rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1 --rdzv-conf=is_host=true
```

不要把该回环地址方案用于跨节点训练。本项目当前验证重点是单机训练。
