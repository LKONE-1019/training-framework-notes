# TorchTitan 分析索引

上游项目：[pytorch/torchtitan](https://github.com/pytorch/torchtitan)。当前笔记基于 [the-fall-moon/torchtitan](https://github.com/the-fall-moon/torchtitan) 的 `kimik3-npu-dev` 分支，结论以文章标注的完整 commit 为准。

| 特性 | 笔记 | 分析基线 | 验证状态 |
| --- | --- | --- | --- |
| 激活重计算 | [FullAC、SelectiveAC 与接入流程](activation-checkpointing.md) | `a8c8a0fcef72396652d2da56fadbb27ff1baa8ec` | 源码静态分析；未运行训练或 GPU/NPU 分布式验证 |

笔记覆盖普通 Trainer 与 Forge 的 AC 接入、Block 包装、SAC 算子保存规则与矩阵乘法计数、MemoryBudgetAC、GraphTrainer 图级重计算，以及当前 K3 NPU 配方的支持边界。

[返回仓库首页](../README.md)
