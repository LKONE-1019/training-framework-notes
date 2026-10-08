# Megatron 分析索引

上游仓库：[NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 与其固定的 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 子模块。

| 特性 | 笔记 | 分析基线 | 验证状态 |
| --- | --- | --- | --- |
| 共卡异构 TP/DP 并行 | [从布局转换到梯度同步](colocated-heterogeneous-parallelism.md) | Bridge `a393057e71aa7289f34b67138352070de333d3e2`；Core `07147d6942dfd4e6d1566b64bb1730f478a1bc32` | 源码静态分析，未运行 GPU 分布式验证 |

笔记覆盖 Core MIMO 的模块配置、rank 与通信组映射、FAN_IN/FAN_OUT/EQUAL 前后向、样本与 token 对齐、按模块 DDP、统一 token 归一化及优化器更新，并说明 Bridge 当前高层配置仍拒绝共卡 rank 重叠的接入边界。

[返回仓库首页](../README.md)
