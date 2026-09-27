# 文本训练数据：选择、下载与接入

适用于本项目约 264M 总参数、151M 激活参数的 MoE，目标是先完整走通预训练 → SFT → 评测，再扩展视觉。核对日期：2026-09-27。

## 1. 先用什么数据

推荐先用 **MiniMind 当前版 mini 组合**完成首轮训练；流程与评测可靠后，扩大预训练数据。8 卡 3090 不要求一开始下载最大数据集。文件 GB 大小不是训练 token 数，也不能据此保证模型能力或训练时长。

| 用途 | 文件 | 下载大小（十进制） |
| --- | --- | ---: |
| 首轮预训练 | `pretrain_t2t_mini.jsonl` | 1.24 GB |
| 首轮指令 SFT | `sft_t2t_mini.jsonl` | 1.74 GB |
| 扩大预训练 | `pretrain_t2t.jsonl` | 8.28 GB |
| 后续扩大 SFT | `sft_t2t.jsonl` | 14.10 GB |

来源：[MiniMind 数据仓库](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)、[数据说明与许可证标签](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/README.md)。当前文件名不同于旧教程的 `pretrain_hq.jsonl`、`sft_mini_512.jsonl`，不要混用。

这套数据容易接入、适合学习小模型流程，但包含合成/蒸馏对话等混合来源；预训练文件不能等同于纯自然网页语料。数据卡同时标注 Apache-2.0 与 CC-BY-NC-2.0，不能理解为全部内容都可以自由商用；后续发布模型或另作他用需核对来源条款。

下载脚本固定版本 `312afb4f76391145c6902f765bb51691c09a12f5`，验证文件大小与 SHA256，保存来源记录。已核对版本元数据并抽查 mini 文件的小段内容；没有在本机完整下载或评估整个语料质量。

## 2. 在服务器下载

以下命令均从项目根目录运行。首轮下载约 2.98 GB；还要存转换结果、分词后的数据。建议数据区域先预留至少 20 GB，再为环境、权重、checkpoint 另留空间，实际需求取决于预处理和保存次数。

```bash
cd ~/university/pku/work/moe-llm-lab
uv sync --locked

# 只查看计划，不下载
uv run --locked python scripts/download_minimind.py --preset mini --list

# 下载两份文件，并验证 SHA256
uv run --locked python scripts/download_minimind.py \
  --preset mini --output data/downloads/minimind-mini-v1
```

不需要 Hugging Face 登录。脚本使用锁定环境中已有的 huggingface_hub，不需要单独安装 CLI 或使用 pip。下载中断可重跑同一条命令；已完成文件会重新校验。已存在但校验失败的文件不会被静默覆盖，先排查磁盘/传输问题，再使用新目录重新下载。

如服务器无法连接 Hugging Face，可尝试第三方镜像（可用性与固定版本是否同步由镜像决定）：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --locked python scripts/download_minimind.py \
  --preset mini --output data/downloads/minimind-mini-v1
```

脚本不发送账户 token，镜像下载也必须通过同一份 SHA256 校验。下载完成目录含两份 JSONL、`DOWNLOAD_PLAN.json` 和 `SOURCES.json`；保留这些记录。

## 3. 转换成本项目的格式

原始 SFT 使用 `conversations`，本项目使用 `messages`。不能只改文件名就直接训练。

```bash
uv run --locked python scripts/convert_minimind.py \
  --pretrain data/downloads/minimind-mini-v1/pretrain_t2t_mini.jsonl \
  --sft data/downloads/minimind-mini-v1/sft_t2t_mini.jsonl \
  --output data/raw/minimind-mini-v1
```

输出目录必须不存在。失败后排查原因，再换新目录重试，不覆盖旧结果。转换按行进行，不把完整 JSONL 一次性加载进内存。

转换策略：

- 预训练输出 `{"text":"..."}`；SFT 输出 `{"messages":[...]}`，保留合法的多轮问答。
- 当前训练普通回答：不使用独立 `reasoning_content`，移除 assistant 内容开头完整的 `<think>...</think>` 块，保留后面的回答；不完整或其他位置的 thinking 标记整条过滤。
- 带工具角色、工具调用元数据或工具标记的对话整条跳过，避免删除工具结果后留下缺失上下文的问答。
- 跳过空内容、不合法角色/轮次和项目保留控制符；无效 JSON 会报错。这里不截断长对话。
- `import-report.json` 记录保留数、过滤原因、实际应用的转换次数和输入/输出/转换器 SHA256。务必查看；目前没有实测整个语料的过滤比例。

原始文件保持不变。此转换只是结构适配，并不代替内容质量审核或跨数据集去重。

## 4. 接上训练指南

回到 [逐步训练指南](training-guide.md) 第 3 步之前，先完成第 0–1 步服务器与 smoke 验收。后续指南命令中的路径统一替换为：

| 指南示例 | 此次实际路径 |
| --- | --- |
| `data/raw/pretrain.jsonl` | `data/raw/minimind-mini-v1/pretrain.jsonl` |
| `data/raw/sft.jsonl` | `data/raw/minimind-mini-v1/sft.jsonl` |

对应 tokenizer 和预处理命令如下，输出路径均应尚不存在：

```bash
uv run --locked moe-lab tokenizer \
  --input data/raw/minimind-mini-v1/pretrain.jsonl \
  --output data/tokenizer-v1.json \
  --vocab-size 16384 --val-ratio 0.05 --seed 42

uv run --locked moe-lab configure \
  --model-config configs/moe-base.json \
  --tokenizer data/tokenizer-v1.json \
  --output configs/model-text-v1.json

uv run --locked moe-lab prepare \
  --input data/raw/minimind-mini-v1/pretrain.jsonl --output data/pretrain-v1 \
  --tokenizer data/tokenizer-v1.json --stage pretrain \
  --max-seq-len 512 --val-ratio 0.05 --seed 42

uv run --locked moe-lab prepare \
  --input data/raw/minimind-mini-v1/sft.jsonl --output data/sft-v1 \
  --tokenizer data/tokenizer-v1.json --stage sft \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
```

检查指南第 4 步的 manifest：实际 token 数、训练/验证数量和 `overlong_sft`。当前 SFT 预处理会跳过超长整段对话，mini 文件也不保证适配我们自己的 512-token tokenizer 窗口。如果大多数被跳过，先尝试 1024 的新预处理目录，并同步修改 SFT 启动的数据路径、做显存短跑；不要直接忽视过滤比例启动长训练。不要宣称下载的全部数据都被用来训练。

正式预训练开始后固定 tokenizer，不因为增加 SFT 或视觉数据重训 tokenizer。保留独立人工评测题，检查预训练与 SFT 的重复、泄漏和中文回答质量；随机留出集不能替代外部能力评测。

## 5. 数据规模如何扩大

首轮验证后，优先试 **8.28 GB 预训练 + 1.74 GB SFT**（约 10.01 GB 下载），暂不必直接上 14.10 GB SFT：

```bash
uv run --locked python scripts/download_minimind.py \
  --preset full-pretrain --output data/downloads/minimind-full-pretrain-v1

uv run --locked python scripts/convert_minimind.py \
  --pretrain data/downloads/minimind-full-pretrain-v1/pretrain_t2t.jsonl \
  --sft data/downloads/minimind-full-pretrain-v1/sft_t2t_mini.jsonl \
  --output data/raw/minimind-full-pretrain-v1
```

需要两份完整文件时使用 `--preset full` 和新的输出目录，合计约 22.37 GB。不要把 mini 与 full 直接拼接，可能重复。继续已有模型训练时保留原 tokenizer；重做独立对照实验时使用独立配置、数据与 runs 目录。

再往后可考虑：

- [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)：其中中文部分约 120B tokens，可作为自然网页预训练数据扩展来源。总仓库体量很大，下一阶段应先选定中文分片与 token 预算，避免整库下载。
- [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)：如 7M_core 子集，适合后续对比指令数据配方；需先在平台接受访问条件，核对 CC-BY-SA-4.0 条款，转换其实际字段。现有脚本专门适配 MiniMind，不保证适用这些数据。

先用同一组固定题对比不同 checkpoint，再决定是否扩容；数据更多不自动意味着回答更好。目前先不下载 DPO 偏好对或 VLM 图片数据。
