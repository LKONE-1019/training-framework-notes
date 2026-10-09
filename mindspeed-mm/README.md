# MindSpeed-MM 分析索引

上游仓库：[Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM)。

| 特性 | 笔记 | 分析基线 | 验证状态 |
| --- | --- | --- | --- |
| 模块异构并行 | [以 Qwen2.5-VL 为例](heterogeneous-parallelism.md) | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` | 静态源码分析，未运行分布式验证 |
| FSDP2 重计算 | [功能、配置与实现链路](fsdp2-recompute.md) | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` | 静态源码分析，未运行训练验证 |

现有笔记分别覆盖 Megatron 模块异构并行，以及插件式 FSDP2 的激活重计算。各篇注明适用后端、配置、源码版本和验证边界。

[返回仓库首页](../README.md)
