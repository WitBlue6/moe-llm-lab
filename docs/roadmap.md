# 学习路线

用户目标已经明确：完整掌握网络搭建、预训练和各种后训练流程。训练资源为 8 卡 RTX 3090。主线采用约 264M 总参数、151M 激活参数的 MoE，避免第一版模型过小，同时为 PPO 多模型训练保留余地。

## 第一阶段：基础链路（本次实现）

- 原生 PyTorch 的 Dense/MoE decoder、RMSNorm、RoPE、GQA、SwiGLU、top-k router 与辅助损失。
- Byte fixture tokenizer、ByteLevel BPE tokenizer 训练、预训练/SFT 数据准备与掩码。
- 预训练、全参数 SFT、AdamW、warmup/cosine、AMP、梯度累积与裁剪、activation checkpointing。
- 单进程与 torchrun DDP、验证 CE/perplexity、采样/KV cache、断点恢复与实验指纹。
- 正确性测试与本地微型合成链路验证。具体记录见 validation.md。

当前不把代码正确性当作已获得语言能力。下一步先在服务器做单卡短跑、两卡/八卡验证，再进行真实语料训练；硬件环境、数据来源/许可证和训练 token 预算待确定。

## 视觉分支（已实现基础代码）

手写 SigLIP 视觉网络：可读取兼容官方预训练参数，或训练随机初始化的图文双编码器（sigmoid 对比损失、跨卡负样本、检索评测、导出视觉 backbone）→ 冻结 SigLIP → projector 对齐 → projector+视觉 LoRA SFT → 图文生成与文本保留评测。支持单机 DDP、恢复和 adapter checkpoint。用户先按 training-guide.md 完成真实文本训练，再启动视觉训练。无图路径关闭视觉 LoRA；统一混合训练为未来对照。尚无正式视觉 checkpoint 或实测能力结论。

## 第二阶段：全参数 SFT 与独立文本 LoRA 对照

基础模型稳定后，用同一领域/任务数据比较 full SFT 与 LoRA。复用已实现的注意力低秩层，另加独立文本 LoRA 训练入口与权重合并，再决定是否适配专家投影。视觉 LoRA 为保留原文本路径不做合并。记录可训练参数、显存、耗时及遗忘情况。

## 第三阶段：DPO（已实现基础链路）

已实现 completion-only 序列 log-prob 求和、冻结 SFT reference、DPO loss、偏好准确率、best 与恢复。下一步使用经过检查的真实偏好数据进行固定问题集验证，不能只凭训练 loss 判断质量。操作见 [RL 训练指南](rl-training-guide.md)。

## 第四阶段：奖励模型 + PPO（已实现基础链路）

已实现独立奖励主干＋标量 head 的 pairwise ranking；PPO 采用共享 actor 主干＋value head。支持可核验奖励/冻结奖励模型、rollout、固定旧概率、reference KL 奖励、GAE、policy/value clipping、entropy、多次更新与有限回答终止。当前每卡完整副本、顺序 rollout，显式梯度归约；不宣称正式模型的训练速度或质量已验收。

首先用可验证任务验证 PPO 数学与数据流，再接入奖励模型。保存独立任务正确率与奖励，区分奖励上升与真实能力改善。actor、critic、reference 和 reward model 的部署方式需依据服务器显存实测，不能按 8×24GB 自动合并显存估算。

## 第五阶段：GRPO（已实现）与算法比较（待实验）

GRPO 已复用 rollout 与奖励接口，实现组内标准化优势、clipped objective 和 reference KL，无 critic。下一步检查组内奖励方差，在匹配任务与采样/更新预算下比较 SFT、DPO、PPO 与 GRPO；不预设任何算法一定更好。

## 训练预算与学习方式

先用短跑测 tokens/s、显存、数据加载及保存耗时，再确定 tokens 与时长。学习项目仍需留出独立评测与固定提示集，但不强求论文新算法。
阅读与实现顺序遵循“模块 → 数学测试 → 小批次过拟合 → 合成端到端 → 真实小样本 → 扩大数据”。

参考：[MiniMind](https://github.com/jingyaogong/minimind)、[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[Tokenizers](https://huggingface.co/docs/tokenizers)。本项目当前没有复制 MiniMind 的源文件。
