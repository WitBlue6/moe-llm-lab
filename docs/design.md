# 第一版设计

## 结构

Decoder-only Transformer，每层采用 pre-norm attention 和 pre-norm FFN，残差连接。Q heads 12、KV heads 4，RoPE 作用于 Q/K，SDPA 使用因果掩码。KV cache 保留较少的 KV heads；初版计算时显式扩展到 Q heads，便于理解与兼容。

每个 token 的 router 在 float32 下得到专家 softmax，选 top-k 后重新归一化权重。只向选中的专家派发有效 token，再加权汇总。没有 capacity 截断、token dropping 或共享专家。按层记录专家分配比例与路由熵。pad token 不参与派发或负载统计。

辅助损失为 `E * sum(mean_assignment_fraction * mean_router_probability)`，在层之间取平均。top-k 的 assignment fraction 总和为 1。训练器按各微批有效输入 token 数加权该辅助损失；它是微批局部负载均衡目标，并非整个全局 batch 上重新计算的非线性均衡目标。

`num_experts=1, experts_per_token=1` 使用无 router 的 Dense SwiGLU。参数量比较必须明确总参数还是激活参数，不能直接用激活参数推断运行速度。

## 监督目标

预训练样本原始 tokens `[BOS, text..., EOS]`，输入 `tokens[:-1]`，标签 `tokens[1:]`。长文档分窗，仅重叠一个边界 token，避免漏掉目标；不拼接不同文档。

SFT 格式为 `[BOS, ROLE, content, EOS, ROLE, content, EOS, ...]`。角色标记、system/user 内容以及它们的 EOS 均不参与监督；assistant 内容与 EOS 参与监督。先构造 mask，再 shift，模型不再二次 shift。长于上限的 SFT 样本整条跳过，并记录数量；不会给被截断回答伪造 EOS。

`token_log_probs` 接受已 shift 的标签，返回逐 token log-prob 与有效位置 mask；这是后续 DPO/PPO 的基础操作。PPO 所需 value head、GAE、rollout buffer、旧策略概率和奖励模型均尚未实现。

## 数据

按内容进行 exact dedup，按带 seed 的内容哈希分配训练/验证集，之后再分窗。BPE tokenizer 训练使用同一切分规则排除验证文档。跨数据集与近似重复内容仍需要额外清理；exact dedup 不代表已完成语义去重。

预处理流式读 JSONL、写 little-endian int32 token/label 对；训练通过 mmap 读取，不将全部 tokens 放入内存。索引和去重哈希集合仍驻留内存，因此超大语料未来需要分片索引和外部去重。当前每条 SFT 对话有独立因果上下文；每条预训练文档窗口也独立，不使用跨文档 packing。

`prepare` 完成后保存 manifest 与二进制/索引 SHA256，加载时校验。输入数据、生成数据与 checkpoint 不应提交 Git。`fixtures/` 是可提交的手写合成样例。

## DDP 与训练

每个 rank 放置完整模型及全部专家，使用 DDP 数据并行，没有 expert parallel。训练 sampler 为 DistributedSampler：数据量不能整除 rank 数时会补齐少量样本；验证按 rank 分片且不补齐。

CE 先求每个微批的有效 token loss sum，收集整个累积组在所有 rank 的有效标签数，再按全局 token 数归一化；乘 world size 抵消 DDP 的梯度平均。避免“每条样本平均后再平均”导致不同 padding/SFT 长度改变优化权重。

初版每个微批都同步，配合 `find_unused_parameters=True` 处理动态未使用专家；不使用 `no_sync` 优化。activation checkpointing 使用非 reentrant 实现。测试比较了两 rank、不等长样本、累积微批与单一全局 batch 的 CE 梯度，并覆盖动态专家与 activation checkpointing。

参数保留 float32，CUDA 前向支持 bf16/fp16 autocast，fp16 使用 GradScaler。梯度非有限时中止，不悄悄推进 schedule 或写入成功 checkpoint。推荐 3090 首先测试 bf16。优化器为 AdamW，norm 参数不做 weight decay，使用 warmup+cosine、梯度裁剪。

恢复保存每 rank RNG 与 sampler epoch/批次位置。在新输出目录恢复，原有实验文件不变。为可复现严格核对配置及训练代码指纹；换代码、数据或并行规模时应使用 init-from 新建实验，而非伪装为精确恢复。

## 已知限制

- 尚无真实语料/正式 checkpoint，也没有模型质量或算法提升结论。
- 主配置未在服务器实测；CPU/Gloo 测试不能证明 CUDA/NCCL 吞吐或数值稳定性。
- 生成 CLI 当前每次一条未 padding 的 prompt；没有批量变长生成、流式 API、beam search 或采样服务。
- 训练不支持 MPS、FSDP、ZeRO、跨卡专家并行、长上下文外推或 fused MoE kernel。
- DPO、PPO、奖励模型、独立文本 LoRA、GRPO 均为后续阶段；视觉 LoRA 已实现。
- perplexity 只在相同 tokenizer/数据/监督位置规则下适合比较；SFT perplexity 不等同通用语言建模 perplexity。


## 视觉扩展

`vision.py` 实现冻结 SigLIP、patch 网格平均池化、LayerNorm+两层 MLP 投影和 VLM 包装。正式配置为 SigLIP base patch16/224，将 14×14 patch 网格池化为 8×8，不增加词表。`inputs_embeds` 与 `input_ids` 二选一进入相同语言主干。

数据处理在首个 user 标记后插入视觉槽位，槽位 ID 为现有 PAD=0，但 attention mask 为 True；真正 batch padding 的 mask 为 False。视觉特征替换槽位 embedding，视觉标签为 ignore，最终 labels 只 shift 一次。超长对话整条跳过。

`lora.py` 包装注意力 Q/V 投影，原始矩阵冻结；低秩 A 随机初始化、B 零初始化。每次 forward 显式传递 adapter_enabled，确保非 reentrant activation checkpoint 的重计算使用同样的模式。未合并 LoRA，无图时返回原 Linear 的计算结果；不会修改 tokenizer、embedding 或输出词表。

`vision_training.py` 复用学习率、AMP、可恢复 batch stream、全局 token 归一化等基础操作。align 只训练 projector，视觉 sft 训练 projector+LoRA；冻结参数不进入优化器。视觉编码器始终 eval/no_grad，但 LLM 不使用 no_grad，以保留损失到投影层的梯度。使用独立视觉 checkpoint 格式，只保存 adapters、训练状态和外部资源指纹，绑定原始文本 SFT checkpoint。

`vision_data.py` 按图片原始文件 SHA256 分 train/val；同图不同问题不能跨 split。相同图片被重新编码后仍可能形成不同哈希，需要外部近重复去重。记录 JSONL 偏移索引与图像 SHA256；训练时逐条读取图像。加载时检查全部图片，启动 I/O 成本在大语料上需要关注。

生成时 prefill 只编码一次图像，后续通过 visual_context=True 保持 LoRA 生效并复用 KV cache。CLI 多轮输入通过 messages-file 提供整个历史，仍传入原图，不能把无图请求误作继续已有图片上下文。

本版本是视觉适配器模式，未实现共享权重混合更新。文本保留通过冻结基座及关闭视觉 LoRA 实现；已验证更新之后的 fp32 文本 logits 与原基座完全一致。视觉效果仍依赖训练好的文本 SFT 基座、真实预训练视觉编码器和图文数据，不能由这个不变性测试推导。
