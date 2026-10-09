# Training Framework Notes

训练框架源码与特性分析笔记，记录 MindSpeed-MM、Megatron、ms-swift、VeOmni 等项目的参数配置、调用流程、通信与数据布局，以及当前实现的支持范围。

## 内容索引

| 框架 / 主题 | 内容 | 状态 |
| --- | --- | --- |
| [MindSpeed-MM](mindspeed-mm/README.md) | [模块异构并行：以 Qwen2.5-VL 为例](mindspeed-mm/heterogeneous-parallelism.md) | 已完成源码静态分析，未运行分布式验证 |
| [MindSpeed-MM](mindspeed-mm/README.md) | [FSDP2 重计算：功能与实现链路](mindspeed-mm/fsdp2-recompute.md) | 静态源码分析，未运行训练验证 |
| [MindSpeed-MM](mindspeed-mm/README.md) | [异步激活卸载机制](mindspeed-mm/async_offload.md) | 已覆盖 legacy 组件与普通 saved-tensor 流程，静态源码分析，未运行设备验证 |
| [Megatron](megatron/README.md) | [共卡异构并行：从 TP/DP 布局转换到梯度同步](megatron/colocated-heterogeneous-parallelism.md) | 已完成源码静态分析，未运行 GPU 分布式验证 |
| [ms-swift](ms-swift/README.md) | 后续补充特性分析 | 待分析 |
| [VeOmni](veomni/README.md) | 后续补充特性分析 | 待分析 |
| [跨框架对比](comparisons/README.md) | 相同特性在不同框架中的实现与限制 | 待补充 |

## 目录结构

```text
training-framework-notes/
├── README.md
├── mindspeed-mm/
│   ├── README.md
│   ├── heterogeneous-parallelism.md
│   ├── fsdp2-recompute.md
│   └── async_offload.md
├── megatron/
│   ├── README.md
│   └── colocated-heterogeneous-parallelism.md
├── ms-swift/
│   └── README.md
├── veomni/
│   └── README.md
└── comparisons/
    └── README.md
```

## 阅读与更新约定

- 每篇笔记注明上游仓库、源码版本或完整 commit、分析日期，以及静态分析或运行验证的状态。
- 代码引用尽量指向固定 commit，并保留文件路径、函数名和行号，方便后续版本对照。
- 区分配置入口、实际执行路径和支持边界；看到参数或通信组初始化，不等于相关训练路径已完整实现。
- 新增结论时注明适用模型、训练后端、并行配置和数据形状假设。
- 性能数据必须附测试条件；没有运行过的配置不标记为已验证支持。

## 单篇分析建议结构

1. 分析范围与源码版本。
2. 特性解决的问题，以及相关概念。
3. 参数设置与生效入口。
4. 按调用顺序介绍实现位置。
5. 具体数据流、张量形状和通信组示例。
6. 反向传播、梯度同步及优化器行为。
7. 当前限制、验证情况与后续阅读位置。

仓库持续增量更新；具体结论以各篇笔记记录的源码版本和验证范围为准。
