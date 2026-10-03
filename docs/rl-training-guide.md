# 从文本 SFT 到 DPO、PPO、GRPO：后训练指南

更新：2026-10-03。目标是理解后训练全过程；默认基座为本项目约 264M 的文本 SFT 模型，单机 8×3090。原生 PyTorch 实现，不依赖 TRL/远程模型/API。当前验收使用微型合成模型和 CPU/Gloo；主模型 CUDA 的显存、速度与质量必须在服务器短跑后判断。

**三种算法从同一 SFT 分别开始，是对照路线，不要求依次 DPO→PPO→GRPO。** DPO 是离线偏好优化；PPO/GRPO 才是在线生成、打分和策略更新。不能把“代码跑通”称为已得到能力提升。

## 0. 同步环境、固定 SFT 基座

同步 `src/`、`scripts/`、`configs/`、`tests/`、`docs/`，整个实验在项目根目录执行。没有新增训练依赖，继续 uv：

```bash
cd ~/moe-llm-lab
uv sync --locked
mkdir -p reports
uv run --locked moe-lab doctor
uv run --locked pytest -q tests/test_posttraining.py tests/test_posttraining_ddp.py
uv run --locked python scripts/smoke.py --output runs/posttrain-smoke-001 --posttrain
uv run --locked moe-lab post-train --help
export TOK=data/tokenizer-v1.json
export RL_RUN=post-v1-20261003
```

选择你的文本 SFT run；下面目录按新曲线中的名称示例，实际不同请改：

```bash
export SFT_RUN=runs/epoch1-20260930-sft
SFT=$(uv run --locked python - "$SFT_RUN" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
if (root/'best.json').is_file():
    path=Path(json.loads((root/'best.json').read_text())['checkpoint'])
else:
    step=json.loads((root/'summary.json').read_text())['step']
    path=root/f'step-{step:07d}.pt'
assert path.is_file(), path
print(path)
PY
)
uv run --locked moe-lab generate --checkpoint "$SFT" --tokenizer "$TOK" --chat \
  --prompt '请用三句话解释训练集和验证集的作用。' --max-new-tokens 160 --temperature 0
```

固定多个未参与训练的问题保存 SFT 基线。若仍严重重复、不能完成基础任务，先改善预训练/SFT。RL 奖励并不能自动修复所有基础能力。

## 1. 数据与奖励选择

### DPO / 奖励模型：偏好对

`data/raw/preferences-v1.jsonl` 每行格式：

```json
{"prompt":"解释为什么需要验证集。","chosen":"验证集用来评估模型在未参与训练的数据上的表现，辅助选择模型和超参数。","rejected":"验证集就是用来直接更新梯度的训练数据。"}
```

多轮用 `messages` 替换 `prompt`，消息数组末尾必须为 user；`chosen` / `rejected` 是最终回答字符串。已有 assistant 历史只提供上下文，损失只覆盖这次完成及 EOS。不要仅凭长度、语气自动判偏好，也不要随意交换标签。

### PPO / GRPO：可核验问题或奖励模型

可核验任务的 `data/raw/rl-prompts-v1.jsonl`：

```json
{"prompt":"计算 17+25，只输出一个数字。","answer":"42"}
```

- `reward="numeric"`：数字完全匹配；接受纯数字、末尾 `#### 数字` 或 `<answer>数字</answer>`。不靠“回答中包含正确数字”给分。
- `reward="exact"`：回答与标签按空白归一化、casefold 后完全相同；适用于明确答案，不能当通用聊天质量奖励。
- `reward="model"`：使用本项目训练的冻结奖励模型。该路线 prompt 数据可以省略 `answer`；分数为 `reward_scale × tanh(raw_score)`，限制奖励幅度。

`answer` 只用于评分，不写入模型 prompt。原数据题目本身若泄漏答案仍需人工清理。相同完整 prompt 在不同偏好对/不同数据准备中，用相同 seed 与 val-ratio 落在同一侧；仅做精确哈希，不保证语义近重复已排除。

### 可选：先生成窄任务合成数据检查流程

没有准备真实数据时，可运行以下命令。它是合成加法数据，不代表真实人类偏好或一般问答评测；想把加法用作正式研究需明确实验范围。

```bash
uv run --locked python scripts/make_rl_arithmetic.py \
  --output data/raw/rl-arithmetic-v1 --count 1000 --seed 42
export PREF_RAW=data/raw/rl-arithmetic-v1/preferences.jsonl
export PROMPT_RAW=data/raw/rl-arithmetic-v1/prompts.jsonl
```

已有真实数据则设置：

```bash
export PREF_RAW=data/raw/preferences-v1.jsonl
export PROMPT_RAW=data/raw/rl-prompts-v1.jsonl
```

以上两组变量设置二选一。不要在合成数据上得到分数后声称通用对齐能力提高。

## 1A. 推荐真实数据：下载与转换

优先先做 MiniMind DPO，再选 PPO/GRPO 的奖励路线。下载脚本仅使用 Python 标准库，固定提交、核对大小和 SHA256，不需要下载模型或安装 datasets；下载中断后重新执行会重新下载该文件，已经校验成功的文件会复用。转换输出目录必须不存在。

| 数据 | 用途 | 下载规模 / 语言 | 来源 |
|---|---|---|---|
| MiniMind `dpo.jsonl` | DPO、训练奖励模型；提取问题用于奖励模型驱动的 PPO/GRPO | 53.7 MB，中英文混合 | [固定版本文件](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/dpo.jsonl) |
| GSM8K 官方 main | 数值奖励 PPO/GRPO；训练 7,473 题、测试 1,319 题 | 约 4.9 MB，英文 | [官方仓库](https://github.com/openai/grade-school-math)、[数据卡](https://huggingface.co/datasets/openai/gsm8k) |

MiniMind 数据卡标有 Apache-2.0 和 CC-BY-NC-2.0，属于混合来源，不能视为统一商业授权；GSM8K 为 MIT。保留来源记录。MiniMind 偏好标签不等于人工逐条审核，建议随机查看样例和过滤比例。这里不使用 `agent_rl_math.jsonl`：它的提示带工具调用设定，当前实现没有工具执行环境，不能直接混作纯数学 RL。

### 下载偏好数据（DPO / 奖励模型）

```bash
uv run --locked python scripts/prepare_rl_sources.py download \
  --dataset minimind-dpo --output data/downloads/minimind-dpo-v1
uv run --locked python scripts/prepare_rl_sources.py convert \
  --dataset minimind-dpo --source data/downloads/minimind-dpo-v1 \
  --output data/raw/minimind-dpo-v1
export PREF_RAW=data/raw/minimind-dpo-v1/preferences.jsonl
```

转换要求 chosen/rejected 共享完整 prompt，最后一轮回答不同；排除工具对话，沿用纯回答 SFT 的 think 清理规则。实测固定版本读取 17,166 对，转换保留 17,137 对，提取 17,021 个不同 prompt（8 对含保留控制 token、21 对答案相同被过滤）；这还不是 tokenizer 长度过滤后的数量。输出 `preferences.jsonl`、去重后的 `prompts.jsonl`、`import-report.json`；prompt 文件不含偏好答案。后续第 2 步还会按 tokenizer 长度过滤和按 prompt 哈希划分训练/验证。

### 路线 A：通用聊天 PPO / GRPO（奖励模型）

```bash
export PROMPT_RAW=data/raw/minimind-dpo-v1/prompts.jsonl
```

先执行第 2 步准备两份数据、第 4 步 DPO（可选），然后执行第 7 步训练奖励模型和 `reward=model` 配置下的 PPO/GRPO。**这份 prompt 没有数值答案，不能直接用第 5、6 步默认 numeric 配置。** 偏好数据与在线问题保持相同 seed/val-ratio，使相同 prompt 位于同一分区。长期实验应额外准备独立问题测试集，避免只评估训练奖励模型自身偏好的分数。

### 路线 B：可核验数学 PPO / GRPO（数值奖励）

```bash
uv run --locked python scripts/prepare_rl_sources.py download \
  --dataset gsm8k --output data/downloads/gsm8k-v1
uv run --locked python scripts/prepare_rl_sources.py convert \
  --dataset gsm8k --source data/downloads/gsm8k-v1 --output data/raw/gsm8k-v1
export PROMPT_RAW=data/raw/gsm8k-v1/train-prompts.jsonl
```

继续第 2、3、5、6 步，保留 `reward=numeric`。转换只把问题写入 prompt，解析 `####` 后的最终数字作为独立 `answer`，去掉数字千位逗号，不把解题过程写入 prompt。**英文 GSM8K 对当前小模型可能偏难；先检查 SFT 能否产生有效数字答案。若组内奖励几乎都为 0，GRPO 难以学习，可先用前面的合成加法检查流程；不要将其成绩当 GSM8K 成绩。** 默认 64 个生成 tokens 是短跑设置，可在正式配置中增大，同时确保 prompt 上限＋生成上限不超过模型 context。

保留官方 test，仅作最终测试；不参与第 2 步、奖励模型训练或调参。准备独立测试目录：

```bash
uv run --locked python scripts/prepare_rl_test.py \
  --input data/raw/gsm8k-v1/test-prompts.jsonl --output data/gsm8k-test-v1 \
  --tokenizer "$TOK" --max-seq-len 448
```

这个目录 train 为空，全部保留样本放在 val，专供 `post-evaluate`，不要用于 `post-train`。训练完成后，将下面 checkpoint 改成实际选定权重；`--max-records 1319` 覆盖全部保留测试题，实际数量和长度过滤见 manifest：

```bash
uv run --locked moe-lab post-evaluate \
  --checkpoint "runs/$RL_RUN-grpo/step-0000100.pt" --base-checkpoint "$SFT" \
  --data data/gsm8k-test-v1 --tokenizer "$TOK" --device cuda --max-records 1319 \
  --output "reports/$RL_RUN-grpo-gsm8k-test.json"
```

路线 A/B 二选一设置 `PROMPT_RAW`，再执行第 2 步；改路线时用新的准备目录和 run 名称。建议在同一测试题、生成预算和奖励定义下比较 SFT/PPO/GRPO，报告保留题数、准确率和生成开销，不预设提升。

## 2. 准备数据、检查过滤

```bash
uv run --locked moe-lab post-prepare --kind preferences \
  --input "$PREF_RAW" --output data/preferences-v1 --tokenizer "$TOK" \
  --max-seq-len 512 --val-ratio 0.05 --seed 42
uv run --locked moe-lab post-prepare --kind prompts \
  --input "$PROMPT_RAW" --output data/rl-prompts-v1 --tokenizer "$TOK" \
  --max-seq-len 448 --val-ratio 0.05 --seed 42
```

preferences 的 512 包含 prompt＋回答＋EOS，过长整个偏好对跳过，不静默截断。prompts 的 448 控制 prompt 长度，后续默认最多生成 64 tokens；训练入口检查 prompt＋生成上限不超过模型 context。若换任务或长度，用新的数据目录。

查看 `manifest.json` 的 `stats`、`split_records`，保留来源、许可证和原始文件版本。输出目录必须不存在；用全局相同 tokenizer，不重新训练 BPE。

## 3. 固定实验配置

模板 `configs/post-{dpo,reward,ppo,grpo}.json` 是起点。复制后在正式训练前调整，不改正在运行的文件：

```bash
export RL_CFG=configs/$RL_RUN
uv run --locked python - "$RL_CFG" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False)
for name in ('dpo','reward','ppo','grpo'):
    c=json.loads((Path('configs')/f'post-{name}.json').read_text())
    with (root/f'{name}.json').open('x') as f: json.dump(c,f,indent=2)
PY
```

| 路线 | 主模型 | 冻结模型 | 默认每个 step 的含义 |
|---|---|---|---|
| DPO | actor 全参数 | 原始 SFT reference | 一次优化器更新 |
| reward | 奖励主干＋标量 head | 无 reference | 一次优化器更新 |
| PPO | actor＋共享主干 value head | SFT reference，可加 reward model | 一批 rollout 后的全部更新 |
| GRPO | actor 全参数，无 critic | SFT reference，可加 reward model | 一批分组 rollout 后的全部更新 |

当前 PPO 的 critic 是共享 actor 主干的 value head，value loss 会更新共享参数；不是独立冻结 critic。训练数据并行每卡完整副本，使用显式梯度归约；没有模型/专家并行、FSDP、vLLM。初始化 fp32 主权重、训练 bf16 autocast，推理按配置精度。先小 batch、短回答验收。

## 4. DPO

单卡短跑：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/dpo.json" \
  --data data/preferences-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-dpo-pilot-1gpu" --stop-after 3
```

八卡短跑：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/dpo.json" \
  --data data/preferences-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-dpo-pilot-8gpu" --stop-after 3
```

确认保存、有限梯度、验证和数据正常后启动新目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/dpo.json" \
  --data data/preferences-v1 --tokenizer "$TOK" --output "runs/$RL_RUN-dpo"
```

公式为 `-log sigmoid(beta × [(logπ(chosen)-logπ(rejected))-(logref(chosen)-logref(rejected))])`，使用完成序列 log-prob **求和**，只监督最终回答和 EOS。训练另有 router aux。reference 始终是原始 SFT，不随 actor 更新。观察 `train_preference_accuracy`、`val_preference_accuracy`、`val_loss`，结合人工检查；偏好指标不等于一般聊天能力。

DPO / reward 按有限 `val_loss` 严格下降保存 best。读取最佳 DPO：

```bash
DPO=$(uv run --locked python - "runs/$RL_RUN-dpo" <<'PY'
import json,sys
from pathlib import Path
p=Path(json.loads((Path(sys.argv[1])/'best.json').read_text())['checkpoint'])
assert p.is_file(),p
print(p)
PY
)
uv run --locked moe-lab post-evaluate --checkpoint "$DPO" --base-checkpoint "$SFT" \
  --data data/preferences-v1 --tokenizer "$TOK" --device cuda --max-records 128
uv run --locked moe-lab generate --checkpoint "$DPO" --tokenizer "$TOK" --chat \
  --prompt '请用三句话解释训练集和验证集的作用。' --max-new-tokens 160 --temperature 0
```

## 5. PPO：先使用可核验奖励

默认 `reward=numeric`，适用于上面的加法数据；其他任务先改配置匹配奖励定义。短跑前不要把所有开放式回答当数字评分。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/ppo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-ppo-pilot-1gpu" --stop-after 3
```

正式八卡（首次八卡建议换 pilot 目录并加 `--stop-after 3`）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/ppo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" --output "runs/$RL_RUN-ppo"
```

采样时记录精确 token IDs、旧策略 log-prob、reference log-prob 和旧 values。每 token 奖励先减 `kl_coef × (old_logp-ref_logp)`，最后一个回答 token 加任务奖励；用 GAE 计算 returns/advantages，全局回答 token whitening，再优化 clipped policy loss、clipped value loss 和可选 entropy。rollout 完成后按 `update_epochs` 重复更新，旧概率保持固定。

采样温度及控制 token 屏蔽在更新时完整重放，不使用 top-k/top-p；只允许文本 token 与 EOS。EOS 与达到 `max_new_tokens` 均在本项目的有限回答任务中视为终止，末端 bootstrap 为 0。记录 `truncated_fraction`，截断高时应重新制定更长预算，不能称已完成答案。

日志 `train_reward` 与 greedy `val_reward` 衡量指定奖励；`reference_kl`、`clip_fraction`、`mean_ratio`、`value_loss` 辅助诊断。训练 objective 不要求单调下降，不能直接把 PPO 的 train_loss 与 SFT CE 比较。

## 6. GRPO

同一个 prompt 采样 `group_size=4` 个回答，按组内均值和总体标准差计算优势；每个回答按自己的 token 长度平均，再对回答平均。使用 PPO 式 clipped objective，另加相对固定 reference 的 k3 KL。无 value head。它是可检查的 GRPO 实现，不宣称精确复现 DeepSeek 的整套配方。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/grpo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-grpo-pilot-1gpu" --stop-after 3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/grpo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-grpo-pilot-8gpu" --stop-after 3
```

通过后正式运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/grpo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" --output "runs/$RL_RUN-grpo"
```

重点检查 `zero_variance_group_fraction`。若长期 1，任务奖励没有组内区分，policy 的奖励优势为 0；仍可能因 KL/aux 更新参数，不能把参数变化当成学到了奖励。应检查任务难度、基座能力、奖励和采样，不盲目扩大步数。

## 7. 可选：训练奖励模型后接入 PPO / GRPO

适用于一般偏好任务；先用同任务 chosen/rejected 训练独立奖励主干＋标量 head，优化 `softplus(score_rejected-score_chosen)`。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked moe-lab post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/reward.json" \
  --data data/preferences-v1 --tokenizer "$TOK" \
  --output "runs/$RL_RUN-reward-pilot-1gpu" --stop-after 3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/reward.json" \
  --data data/preferences-v1 --tokenizer "$TOK" --output "runs/$RL_RUN-reward"
```

读取 reward best，先检查留出偏好准确率，不能仅凭 loss 使用：

```bash
RM=$(uv run --locked python - "runs/$RL_RUN-reward" <<'PY'
import json,sys
from pathlib import Path
p=Path(json.loads((Path(sys.argv[1])/'best.json').read_text())['checkpoint'])
assert p.is_file(),p
print(p)
PY
)
uv run --locked moe-lab post-evaluate --checkpoint "$RM" --base-checkpoint "$SFT" \
  --data data/preferences-v1 --tokenizer "$TOK" --device cuda --max-records 128
uv run --locked python - "$RL_CFG" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('ppo','grpo'):
    c=json.loads((root/f'{name}.json').read_text()); c['reward']='model'
    with (root/f'{name}-rm.json').open('x') as f: json.dump(c,f,indent=2)
PY
```

启动新的 PPO-RM 实验，GRPO 将配置改为 `grpo-rm.json` 即可：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/ppo-rm.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" --reward-checkpoint "$RM" \
  --output "runs/$RL_RUN-ppo-rm"
```

奖励模型路线每卡多一份冻结模型，先单卡新目录短跑测峰值显存。真实一般聊天需要一般聊天偏好数据和相同分布的 prompts，不能用加法 reward model 给所有领域打分。任务奖励增加也可能是奖励投机，保留独立人工/规则评测。

## 8. 评测与生成

PPO/GRPO 不伪造 CE `val_loss`，记录 `val_reward`；当前保留定期/最终 checkpoint，没有基于 RL train_loss 的 best。选中实际存在的文件后评测同一固定问题集：

```bash
export POLICY="runs/$RL_RUN-grpo/step-0000100.pt"
test -f "$POLICY"
uv run --locked moe-lab post-evaluate --checkpoint "$POLICY" --base-checkpoint "$SFT" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" --device cuda --max-records 128 \
  --output "reports/$RL_RUN-grpo-eval.json"
uv run --locked moe-lab generate --checkpoint "$POLICY" --tokenizer "$TOK" --chat \
  --prompt '计算 17+25，只输出一个数字。' --max-new-tokens 64 --temperature 0
```

上述 step100 假设保留模板总步数；epoch 模式或 stop-after 后要换成实际 step。reward=model 评测也传同一 `--reward-checkpoint "$RM"`。独立 `post-evaluate` 使用 `--max-records`，不支持 `--eval-max-batches`。

留出 val 用于调参；最终验收另备冻结的 test 问题，用新数据目录的 val 侧作为显式评测输入，记录其准备方法。不能不断查看同一 test 调参后仍称独立测试。建议同题比较 SFT、DPO、PPO、GRPO 的质量、长度、奖励、KL 和成本，而不是只比较单次输出。

策略 checkpoint 兼容现有 `moe-lab generate/evaluate`；奖励模型 checkpoint 不作为聊天模型。普通 `evaluate` 还可用原文本 SFT 验证集检查 CE 遗忘，保持相同精度与数据范围。

## 9. epoch、恢复与输出

- `post-train --epochs N` 覆盖配置 max_steps。DPO/reward 一轮步数按 prompt/pair 数和 batch/累积计算；PPO/GRPO 一轮表示把 prompt 采样器走完一次，不是每条生成回答只更新一次。
- PPO/GRPO 一个日志 step 是一批 rollout，含 `update_epochs` 次优化器更新；累计看 `optimizer_updates`。`grad_accum_steps` 控制每批收集的 prompt 微批数，online 更新顺序逐回答反传；不扩大单次微批 forward。
- GRPO 全局每批回答数最多为 `卡数×batch_size×grad_accum_steps×group_size`，不要把它与 DPO 样本预算当同一计算量。
- `--stop-after` 是全局日志 step，只提前停止，不改原 LR 计划。DPO/reward 会在每次验证结果改善时额外保存；online 算法按定期/结束保存。

仅中断后恢复，文件必须存在，保留相同 base、config、data、tokenizer、reward、world size、代码和环境，换新输出：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --locked torchrun \
  --standalone --nnodes=1 --nproc-per-node=8 -m moe_llm.cli post-train \
  --base-checkpoint "$SFT" --config "$RL_CFG/grpo.json" \
  --data data/rl-prompts-v1 --tokenizer "$TOK" \
  --resume "runs/$RL_RUN-grpo/step-0000020.pt" \
  --output "runs/$RL_RUN-grpo-resumed"
```

epochs 模式恢复须重复同样的 `--epochs N`。新版会改变代码指纹，旧文本/VLM 训练仍用原代码精确恢复；已完成 SFT 可作新版 base。此入口不支持中途改预算，也不提供从其他算法 checkpoint 直接 init-from 的捷径。

每个 run 保存 `run.json`、`metrics.jsonl`、`summary.json`、optimizer/各 rank RNG/epoch-cursor checkpoint。在线策略按卡保留 `rollouts-rankXXX.jsonl` 生成轨迹；记录全局训练生成 token、prompt 曝光次数、rollout 数、optimizer 更新次数、时间、分配 GPU 秒与峰值显存。生成 token 计数仅指训练 rollout，不包含验证；时间开销包含本次运行内验证与保存。没有外部模型 API 调用。图表脚本可继续读取 train/aux；online 无 val_loss 的面板显示未记录，reward 指标直接读 JSONL。

RL 改变了文本基座权重，因此已有视觉 adapter 不能直接绑定新的 RL checkpoint：它记录原 SFT 的 SHA256。继续使用旧 VLM 基座，或针对新的文本基座另建视觉训练实验；不要删除绑定校验。

## 数学与实现参考

- [DPO 原论文](https://arxiv.org/abs/2305.18290)：离线偏好对与冻结 reference。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：旧策略概率比与 clipped policy objective。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：组相对优势与无 critic 的 GRPO。

本项目使用这些算法思想实现教学训练器；没有复制其完整训练系统，也没有实测证明优于参考项目。
