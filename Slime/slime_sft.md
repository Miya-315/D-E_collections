# Slime SFT


**注意**：本文档以 `slime` 仓库中的实现为准，示例配置参考 `run-glm4.7-flash-sft.sh`中的 `SFT_ARGS`。

## 目录
- Slime SFT 与常规模型微调的区别
- 训练流程
- 关键参数说明
- 示例配置片段
- 优点与实践建议

## Slime SFT 与其他 SFT 的主要区别

- 复用 RL 框架的数据生成模块：
  - `slime` 将 SFT 视为一种特殊的 rollout 流程，通过 `--rollout-function-path slime.rollout.sft_rollout.generate_rollout` 指定从文件读取数据的生成函数，从而复用同一套数据/rollout 接口。
  - 这样可以无缝地把 SFT → RL 的流程串联起来，便于后续切换。
- 训推一体、Megatron + SGLang 架构：
  - 训练端使用 Megatron-LM，推理端原生集成 SGLang（RL 场景才需要）。SFT 阶段通常不启动 SGLang，但框架保持一致性。
- 更灵活的多轮/工具调用支持与 mask 生成：
  - 使用 `MultiTurnLossMaskGenerator` 等工具自动生成多轮对话的 loss mask，支持包含 tools 的样本处理。
- 可配置的损失粒度：
  - 支持 `--calculate-per-token-loss`（按 token 平均）以匹配常见的 SFT 惩罚方式，而不是默认的 RL per-sample 统计方式。

## 训练流程（按步骤）

1. 初始化与资源调度
   - 创建 Ray placement groups，初始化 rollout manager 与训练模型（Megatron actor/critic）。
   - SFT 常用标志：`--debug-train-only`（只加载 Megatron，不初始化 SGLang）。

2. 数据准备
   - 数据通常为 JSONL/Parquet，每行包含 `messages` 等字段（由 `--input-key` 指定）。
   - 使用 `slime` 的 `Dataset`/`DataSource` 读取并分批。

3. Rollout（SFT 特殊实现）
   - 通过 `slime.rollout.sft_rollout.generate_rollout` 将原始样本转换为训练所需的 `Sample`：
     - 加载 tokenizer/processor（`load_tokenizer`、`load_processor`）
     - 生成 `token_ids`、`loss_mask`、`response_length` 等字段
     - 将 `reward` 置 0（SFT 无 reward）

4. 训练主循环（ `train_async.py`）
   - 从 rollout manager 获取 batch（可能是同步或异步），调用 `actor_model.async_train()` 执行训练步骤。
   - 每步前向 -> 仅对未被 mask 的 token 计算 loss -> 反向 -> 优化器更新。

5. 存储与周期性评估
   - 按 `--save-interval` 保存 checkpoint；可在训练中调用 rollout 的 eval 方法进行评估。

## 关键参数说明（常用）

- `--rollout-function-path`: 指定数据生成函数（SFT 使用 `slime.rollout.sft_rollout.generate_rollout`）。
- `--prompt-data`: 数据文件路径（JSONL/Parquet）。
- `--input-key`: 从样本中读取文本字段的 key，常为 `messages`。
- `--num-epoch`: 遍历数据的次数。
- `--rollout-batch-size` / `--global-batch-size`: SFT 推荐二者相同（读一批训一批）。
- `--loss-type sft_loss`: 使用 SFT（交叉熵）损失类型。
- `--calculate-per-token-loss`: 按 token 平均损失（SFT 常用）。
- `--disable-compute-advantages-and-returns`: 跳过 RL 特有的 advantage/return 计算。
- `--debug-train-only`: 只初始化训练端（Megatron），便于本地调试 SFT。

## 示例配置片段（参考）

```bash
SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data /path/to/data.jsonl
  --input-key messages
  --num-epoch 3
  --rollout-batch-size 128
  --global-batch-size 128

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
```

## 优点与实践建议

- 平滑过渡到 RL：同一套框架支持 SFT 和 RL，参数层面切换即可。
- 高性能训练：利用 Megatron 的并行能力（TP/PP/EP/CP）进行大规模训练。
- 灵活的数据接口：通过 custom rollout 可以轻松加入工具调用、验证器或多模态输入。

实践建议：
- SFT 初始阶段可使用较小 batch 与 `--debug-train-only` 调试数据 pipeline 与 loss mask。
- 确认 `--rotary-base`、tokenizer 与 checkpoint 配置一致以避免转换问题。

## 总结

`slime` 中的 SFT 并非简单的独立训练脚本，而是被设计为 RL 框架下的一种特殊 rollouts 配置。它将传统的监督微调流程与高性能训练、灵活的数据生成机制结合，便于后续在同一套基础设施上开展 RL 实验。
# 1. slime SFT 流程

###  SFT数据流动全流程

1. 数据准备 
有一堆 jsonl 文件（如 math_with_tools_slime_format.jsonl），每一行是一个对话样本，包含 prompt、messages、tools 等字段。
这些数据通常已经过预处理，格式统一，便于后续加载。
1. 数据加载（slime/utils/data.py 的 Dataset 类）
训练脚本会通过 Dataset 类加载这些 jsonl 数据。
Dataset 会把每一行 jsonl 解析成 Python 字典，进一步处理成模型需要的格式（如拼接 prompt、处理 tools、生成 token 等）。
这里可以插入 DeepSeek encode 逻辑（如 use_deepseek_encode=True），让数据自动变成 DeepSeek 风格的 prompt。
1. 数据分发（slime/rollout/data_source.py）
DataSource（如 RolloutDataSource）负责把 Dataset 加载好的数据分批（batch）送给训练主循环。
每次训练迭代，DataSource 会“取出”一批样本，供模型学习。
1. Tokenizer/Processor
在 Dataset 初始化时，会加载 tokenizer（分词器）和 processor（多模态处理器）。
Tokenizer 把文本 prompt 转成模型能理解的 token id（数字序列）。
Processor 负责图片等多模态数据的预处理（如果有）。
1. 掩码生成（mask_utils.py）
训练时需要告诉模型哪些 token 需要算 loss（即哪些是“答案”）。
掩码生成逻辑会根据 prompt/messages 结构，自动生成 loss mask，保证模型只在需要的地方学习。
1. 训练主循环
训练脚本（如 train_async.py）会不断从 DataSource 取 batch，送入模型。
模型前向推理，计算 loss（损失），反向传播，更新参数。
训练过程中会自动 shuffle（打乱）数据，提升泛化能力。
1. 保存与恢复
训练到一定步数会保存 checkpoint（模型快照）。
可以随时恢复训练，继续学习。

| 阶段                    | 时间          | 进程                    | 关键事件                                 | 技术细节                                              |
| :-------------------- | :---------- | :-------------------- | :----------------------------------- | :------------------------------------------------ |
| **资源调度**              | 09:48:15    | Placement Group       | 创建 8 GPU 资源组                         | Ray 集群初始化，绑定节点 `10.244.195.16`                    |
|                       |             |                       | ⚠️ Triton 不支持，回退 CPU                 | `fla/utils.py:215`，可能影响 attention 性能              |
| **RolloutManager 启动** | 09:48:16    | `pid=1249960`         | 检测 Megatron Core + FSDP              | 使用 Megatron-FSDP 混合并行策略                           |
|                       | 09:48:17    |                       | **启动 SGLang Router**                 | 端口 `:3492`，`backend='sglang'`                     |
|                       |             |                       | 配置参数                                 | `policy='cache_aware'`，`history_backend='memory'` |
| **数据加载**              | 09:48:20    |                       | ✅ **加载 SFT 数据集**                     | `math_with_tools_slime_format.jsonl`              |
|                       |             |                       | 样本数：**23,329** 条                     | 格式：带 tool 的 math 数据                               |
|                       |             |                       | 样本结构初始化                              | `status=PENDING`，`response=''`，`reward=None`      |
| **函数导入**              | 09:48:22    |                       | 导入 rollout 生成函数                      | `slime.rollout.sft_rollout.generate_rollout`      |
|                       |             |                       | 导入评估函数                               | `eval_generate_rollout`（同名）                       |
| **训练 Actor 初始化**      | 09:48:32    | `pid=1250342`         | Rank 0 启动                            | 应用 `torch.distributed` monkey patch               |
|                       | 09:48:44-45 | `pid=1250647/1250653` | Rank 1-2 启动                          |                                                   |
|                       |             | ...                   | Ranks 3-7 启动（共 8 个）                  |                                                   |
| **分布式组网**             | 09:48:45    | Rank 0                | `[Gloo] Rank 0 connected to 7 peers` | TP=8（Tensor Parallel）确认                           |
| **硬件绑定**              | 09:48:45    | All Ranks             | `Set NUMA affinity for GPU 0-7`      | 每个 Actor 绑定独立 GPU                                 |
| **Tokenizer 加载**      | 09:48:45    | All Ranks             | ⚠️ Legacy tokenizer 警告               | 建议迁移到 `megatron.core.tokenizers`                  |
| **模型初始化**             | 09:48:45    | Rank 0                | `building HuggingFaceTokenizer`      |                                                   |
|                       |             |                       | `setting random seeds to 1234`       | 确定性训练                                             |
|                       |             |                       | **参数量：3.8B**                         | `tensor, pipeline rank (0,0): 3,805,411,328`      |
|                       |             |                       | 其他 Ranks：3.78B                       | `rank (2,1): 3,782,175,488`                       |
| **Checkpoint 加载**     | 09:49:11    | All Ranks             | 加载 sharded\_state\_dict metadata     | `fully_sharded_model_space`                       |
|                       |             |                       | ⚠️ `allow_shape_mismatch` 参数         | 非标准参数，兼容处理                                        |
|                       | 09:49:47    |                       | **加载分布式检查点**                         | `GLM-4.7-Flash_slime/`                            |
|                       |             |                       | 恢复迭代：**1999**                        | 从 checkpoint 继续训练                                 |
|                       |             |                       | ⚠️ `load_state_dict` deprecated      | 建议改用 `load` API                                   |
|                       |             |                       | ⚠️ ShardedTensor → DTensor           | PyTorch 分布式演进警告                                   |
| **训练就绪**              | ~09:49:48   |                       | 所有 Ranks 完成初始化                       | 准备进入 SFT 训练循环                                     |
