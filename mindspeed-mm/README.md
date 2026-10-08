# MindSpeed-MM 分析索引

上游仓库：[Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM)。

| 特性 | 笔记 | 分析基线 | 验证状态 |
| --- | --- | --- | --- |
| 模块异构并行 | [以 Qwen2.5-VL 为例](heterogeneous-parallelism.md) | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` | 静态源码分析，未运行分布式验证 |

当前笔记围绕 Megatron 训练路径，覆盖通信状态、图像输入输出重分配、数据加载、microbatch 缩放、异构流水线调度及反向支持边界。

[返回仓库首页](../README.md)
