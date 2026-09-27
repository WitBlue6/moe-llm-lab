# 本地验证记录

日期：2026-09-27。环境：macOS arm64 / Apple M4、Python 3.10.17、PyTorch 2.14.0、Transformers 4.57.6、Pillow 12.3.0、CPU fp32。依赖由 uv.lock 管理。

## 自动测试

完整命令：`uv run --locked --extra vision pytest -q`。

在最终项目目录运行结果：**32 passed in 25.64s**，涵盖文本、视觉、诊断 CLI 和两个进程的 DDP。

文本测试保留了：因果性、Dense/MoE 参数量、主配置 meta 参数量、KV cache 对照、pad/router 统计、稀疏派发与 router 梯度、空专家、checkpointing 梯度、CE/log-prob mask、单批次过拟合、SFT 多轮 mask、数据切分/去重/指纹、BPE 往返与验证集隔离、单进程和两进程 DDP 恢复。

新增视觉测试覆盖：

- `inputs_embeds` 与原 input_ids 路径输出一致，且梯度可回传到 embedding 输入。
- align/sft 更新后，冻结的基座和视觉参数不变；无图 logits 与原 LLM 逐项一致。
- 投影层能够通过冻结的 LLM 获得非零梯度；视觉 LoRA 在 SFT 中实际更新。
- 显式 LoRA 开关在 activation checkpoint 的反向重计算中保持一致，梯度与无 checkpoint 对照一致。
- 图文完整前向与缓存解码一致；生成时视觉编码器只运行一次。
- 图像槽位范围、占位 ID 和 attention mask 校验。
- 同图不同问题按图像内容分到同一 split；视觉标签 ignore、真实视觉 mask、pad mask 与图像篡改检测。
- 使用本地随机初始化的小型 **SiglipVisionModel** 保存/加载真实 safetensors 文件，并验证 SiglipImageProcessor 和 patch pooling 接口。没有下载或测试正式预训练 SigLIP 权重。
- 视觉 align → sft → 精确恢复；adapter checkpoint 不含原 LLM/专家权重；拒绝错误基座。
- 两进程 CPU/Gloo 的视觉对齐、视觉 SFT、checkpoint 保存与恢复；恢复参数与连续训练逐项一致。
- doctor 命令成功退出，configure 使用实际 tokenizer 词表生成新配置。

本机 torchrun 默认检测到不可解析主机名，因此测试显式使用 127.0.0.1 rendezvous/local address 和回环接口，不修改系统网络设置。限制回环通信的沙箱需要开放该本机测试能力。

## 完整 CLI smoke

已实际执行：

```bash
uv run --locked --extra vision python scripts/smoke.py --output <新的测试目录> --vision
```

流程为：文本数据准备 → 微型预训练 8 steps → 微型文本 SFT 4 steps → 文本生成 → 生成合成 PNG → 图文数据准备 → 视觉对齐 4 steps → 视觉 LoRA SFT 4 steps → 图文/遮图/文本保留评测 → 图文生成。

实际结果：所有步骤成功；`text_logits_identical: true`、`text_max_logit_error: 0.0`。测试输出和权重保存在临时目录，没有把这些权重当作正式模型交付。

图像为程序生成的色块，fixture 视觉编码器为随机冻结的 patch convolution，LLM 也只做极少步合成训练。因此生成不代表识图、算术或聊天能力；loss 与遮图差异不能用来报告泛化或算法提升。

## 尚未验证

- 用户服务器上的 CUDA 13 驱动兼容性、NCCL、bf16/fp16、8 卡显存和吞吐。
- 264M 主模型的真实语料预训练/SFT质量。
- 正式预训练 SigLIP 权重与真实图文语料上的视觉学习效果。
- 图文混合更新共享权重、DPO、PPO、奖励模型、独立文本 LoRA 和 GRPO；这些尚未实现。

开始正式训练前，按 training-guide.md 的环境诊断与 GPU 短跑逐项验收。
