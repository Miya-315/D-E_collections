## Megatron-LM 架构
Megatron-LM 是由 NVIDIA 开发的开源框架，专门用于大规模语言模型的高效分布式训练 。
核心功能
分布式训练并行策略
张量并行（Tensor Parallelism）：将单层 Transformer 拆分到多个 GPU
流水线并行（Pipeline Parallelism）：将不同层分配到不同 GPU
序列并行（Sequence Parallelism）：针对长序列的优化
数据并行（Data Parallelism）：在多个节点间复制模型 
Transformer Engine 集成
支持 FP8 精度训练（Hopper/Blackwell GPU）
融合 FlashAttention 内核
通信-计算重叠优化 
训推统一支持
包含专门的 Inference System（推理系统）
支持静态/动态批处理、KV Cache 分块、CUDA Graphs 加速 
架构定位
主要用于训练端，但也包含推理能力，是 NVIDIA GPU 上训练大模型的行业标准框架 
## SGLang 架构
SGLang 是一个高性能的 LLM 推理服务框架，由 LMSYS 团队开发，核心创新在于将 LLM 交互视为结构化程序而非独立请求 。
核心创新
- RadixAttention 机制
使用基数树（Radix Tree）管理 KV Cache
实现前缀缓存（Prefix Caching）：自动复用共享前缀的计算
支持层级缓存，适用于多轮对话、ReAct 代理等场景 
- Cache-Aware 调度器
优先调度与当前 GPU 缓存数据共享前缀的请求
避免缓存抖动（Cache Thrashing）
最大化批量处理效率 
- 结构化生成
前端 DSL（领域特定语言）定义复杂工作流
后端运行时优化多调用、分支、循环等控制流
相比 vLLM 在复杂工作负载上可达 5× 吞吐量提升 
分离式服务（Disaggregated Serving）
将 Prefill（计算密集）和 Decode（内存密集）阶段分离到不同 Worker
独立扩缩容，通过 RDMA 高效传输 KV Cache 
