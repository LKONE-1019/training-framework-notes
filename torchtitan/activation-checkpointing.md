# TorchTitan 激活重计算：FullAC、SelectiveAC 与接入流程

[返回 TorchTitan 索引](README.md) · [返回仓库首页](../README.md)

TorchTitan 在本次分析版本中提供三种普通训练 AC 策略：`FullAC` 按 Transformer Block 重计算，`SelectiveAC` 在 Block 内按算子缓存或重算，`MemoryBudgetAC` 将取舍交给编译器。实验目录的 GraphTrainer 另有基于 FX 图的重计算与激活卸载流程。**框架存在这些机制，不等于每个模型、融合算子和设备后端都已经适配。** 当前 K3 NPU debug 配方明确关闭 AC。

| 项目 | 分析范围 |
| --- | --- |
| 上游项目 | [pytorch/torchtitan](https://github.com/pytorch/torchtitan) |
| 实际分析仓库 | [the-fall-moon/torchtitan](https://github.com/the-fall-moon/torchtitan)，分支 `kimik3-npu-dev` |
| 完整源码 commit | `a8c8a0fcef72396652d2da56fadbb27ff1baa8ec` |
| 分析日期 | 2026-10-16 |
| 主线 | 普通 `Trainer`、Qwen3 非 PP 接入；补充 Forge、K3 NPU 和 GraphTrainer 差异 |
| 并行边界 | 调用链以 PP=1 为主；示例为单卡、DP=TP=CP=PP=EP=1，不推导任意多卡组合的可用性 |
| 验证状态 | 源码静态分析；检查配置、调用链和文档链接，未运行训练、GPU/NPU 性能或分布式数值验证 |
| 外部依赖核对 | 阅读本机 PyTorch `2.7.1+cpu` 的 checkpoint 与 wrapper 源码，版本 commit 为 `e2d141dbde55c2a4370fac5165b0561b6af4798b`；仅用于解释基础机制，不代表其满足该 TorchTitan 分支的依赖要求 |

源码工作区的已跟踪文件与上述 commit 一致；已有未跟踪材料未作为实现依据。TorchTitan 源码引用固定到 fork 的完整 commit，不把当前 fork 的配置结论推广到上游最新主分支。外部 PyTorch 行为有版本差异，尤其是策略回调时机、`early_stop` 参数和 CPU offload 扩展。

## 1. AC 解决什么问题

Activation Checkpointing 通常译为“激活检查点”，训练中也常称“激活重计算”。正常自动求导需要保存部分前向中间张量，供反向计算梯度使用。AC 缩减长期保存的中间激活，反向需要时从保留的输入、参数等信息重新生成。

```text
构建模型时：给选中的 Block 安装 checkpoint 包装

训练前向：使用当前 microbatch 输入计算 loss，保留部分恢复所需信息
训练反向：按需重新执行前向计算，恢复中间激活，再继续计算梯度
优化器：在梯度累积和裁剪等步骤之后执行正常更新
```

AC 不重新读取 batch，不增加一次 optimizer step，也不是把权重保存到磁盘的模型 checkpoint。它与 FSDP 分工不同：AC 管理中间激活，FSDP 管理参数与梯度分片及通信。重算仍然需要参数，不能认为打开 AC 就不再发生参数聚合。

## 2. 三种策略及配置方式

策略类型集中定义在 [`ActivationCheckpointingConfig`][ac-types]。

| 配置 | 工作方式 | 主要代价与边界 |
| --- | --- | --- |
| `None` | 不安装普通 AC | 不额外引入 checkpoint 重算 |
| `FullAC.Config()` | 对每个被包装 Block 重新执行前向 | 通常节省更多激活内存，增加较多计算；仍要保留输入等必要信息 |
| `SelectiveAC.Config()` | 对 Block 内的算子选择缓存输出或重算 | 固定规则折中；不是实时测量每个算子的耗时后再选择 |
| `MemoryBudgetAC.Config(memory_budget=0.5)` | 设置编译器的激活内存预算 | 要求开启模型编译；不是直接给每层安装 FullAC 包装 |

可以在模型配方中选择其中一种。以下为配置写法示意，不是对当前 K3 NPU 可运行性的声明：

```python
config.activation_checkpoint = SelectiveAC.Config()
```

如果需要预算策略，还必须满足 `compile.enable=True` 且 `compile.components` 包含 `"model"`；[Trainer 配置校验][trainer-budget-check]会检查这两个条件。

[普通 Trainer 默认配置][trainer-default]使用 `SelectiveAC.Config()`，具体模型配方可以覆盖它。例如 [Qwen3 debug][qwen-recipe]显式选择 SAC，而 [K3 debug][k3-config]显式设置 `None`。配置加载顺序为：模型 `config_registry` 函数生成基准配置，随后 [`ConfigManager.parse_args()`][manager]用 CLI 参数覆盖。联合类型定义的策略分支名为 `selective`、`full`、`memory-budget`、`none`；不是旧式的 `mode="selective"` 字段。

公共配置包括：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `preserve_rng_state` | `True` | 向 checkpoint 传递随机状态保存要求，用于重算一致性；后端支持仍需核实 |
| `determinism_check` | `"default"` | 使用 PyTorch 的默认重算一致性检查；不能当作完整数值对齐测试 |
| `debug` | `False` | 请求额外 checkpoint 调试信息，组合支持依赖 PyTorch 版本 |
| `force_recompute_mm_shapes_by_fqns` | `["moe.router.gate"]` | SAC 专用：通过模块名称收集需要优先重算的权重形状 |
| `memory_budget` | `0.5` | MemoryBudgetAC 专用：编译器预算控制量，范围 `[0,1]` |

## 3. 从配置到模型包装：代码跳转主线

`ModelSpec` 是模型接入信息的集合，包含结构配置、并行化函数、流水线函数和权重适配器等。**`model_spec.model` 是配置，构建出的 `model` 才是实际 `nn.Module`。** Qwen3 将 `parallelize_fn` 绑定为 `parallelize_qwen3`，Trainer 据此调用模型专用逻辑。

| 顺序 | 代码入口 | 完成的工作 |
| --- | --- | --- |
| 1 | [`qwen3_debugmodel()`][qwen-recipe] | 选择 `SelectiveAC.Config()`，把 `ModelSpec` 放入训练配置 |
| 2 | [Qwen3 `model_registry()`][qwen-spec]、[`ModelSpec`][model-spec] | 绑定 `parallelize_fn=parallelize_qwen3` |
| 3 | [`train.main()`][entry] | 解析配置，通过 `config.build()` 构建 Trainer |
| 4 | [Trainer 调用 `parallelize_fn`][trainer-parallelize] | 将 `config.activation_checkpoint` 作为 `ac_config` 传下去 |
| 5 | [`parallelize_qwen3()`][qwen-parallelize] | 执行 `ac_config.build(...).apply(model)`；之后按配置编译，再应用 FSDP |
| 6 | [`Configurable.Config.build()`][config-build] | 用配置绑定的 `_owner` 创建 `SelectiveAC`、`FullAC` 或 `MemoryBudgetAC` 实例 |
| 7 | [`ActivationCheckpointing.apply()`][ac-apply] | FullAC/SAC 遍历 `model.layers`，调用 `_wrap_block()`，把包装模块注册回原层位置 |
| 8 | [`FullAC._wrap_block()`][full-ac] / [`SelectiveAC._wrap_block()`][selective-ac] | 创建 checkpoint 包装；SAC 额外传入策略上下文 |
| 9 | [`Trainer._forward_backward_body()`][forward-backward] | 正常调用模型、loss 和 `loss.backward()`，checkpoint 在反向恢复所需激活 |

关键接入口只有这一段：

```python
if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)
```

`build()` 创建的是策略对象，`apply()` 才将策略安装到模型上。FullAC/SAC 的基类 `apply()` 遍历所有直接位于 `model.layers` 下的 Block；当前没有通用配置列表用来指定“只包装第 3 层”或“只包装 Attention”。若要按模块选择范围，需要扩展安装逻辑，而不是往矩阵乘法形状配置中填层名。

非 PP 分支完成并行化后返回 Trainer，继续分配目标设备存储并初始化权重。**此时只完成安装，还没有运行前向或触发重算。** PP 分支会把 AC 配置传入流水线构建逻辑，本文不展开其调度与 microbatch 生命周期。

### ForgeEngine 的接入位置

Forge 是另一条引擎入口，不是 `Trainer` 返回后必经的一站。其 [配置字段和预算校验][forge-config]与普通 Trainer 类似；非 PP 分支在 [`ForgeEngine` 调用 `parallelize_fn`][forge-parallelize]时同样传入 `ac_config`，随后复用模型的 AC 接口。

## 4. FullAC 与 SelectiveAC 如何包装 Block

`ptd_checkpoint_wrapper` 是 PyTorch `checkpoint_wrapper` 的导入别名。函数接收原模块，返回持有该模块的 `CheckpointWrapper`，后者仍是 `nn.Module`，不是复制一份模型参数。本机核对的 [PyTorch wrapper 实现][pt-wrapper]会把普通调用转为 checkpoint 调用，默认使用非重入实现。

```text
原来：model.layers[i](x) → 原 Block.forward(x)

包装后：model.layers[i](x)
          → CheckpointWrapper.forward(x)
          → checkpoint(原 Block, x, ...)
          → 正常前向 / 反向需要时恢复激活
```

FullAC 直接创建包装，SAC 额外传入：

```python
context_fn=lambda: create_selective_checkpoint_contexts(
    _get_custom_policy()
)
```

策略函数负责决定算子如何处理，PyTorch 上下文负责缓存和读取输出。SAC 保存的是算子对外返回的结果，不代表保存融合算子内部所有中间张量；即使全部选择保存，也不等于普通 autograd 的最小保存集合。

本分支 FullAC 与 SAC 都显式传入 `early_stop=False`，意图让 checkpoint 重算区域完整执行。该参数依赖目标 PyTorch 的接口支持；本文核对的本机旧版 PyTorch 仅用于说明 wrapper 基础结构，不能据此声称整个 TorchTitan 分支可直接在该旧版上运行。

## 5. SelectiveAC：按优先级解释选择策略

策略实现位于 [`wrapped_policy()`][sac-policy]，它返回 `CheckpointPolicy`。普通 SAC 主要用到 `MUST_SAVE` 和 `PREFER_RECOMPUTE`；`MUST_*` 表达不应被其他子系统覆盖的要求，`PREFER_*` 表达偏好。没有“按概率重算”的含义。

### 5.1 首先保留 CUDA 到 CPU 的拷贝输出

满足 `_to_copy`、源设备为 CUDA、目标设备为 CPU 时，直接返回 `MUST_SAVE`。例如前向把 MoE 元数据搬到 CPU，缓存的是搬运后的 CPU 张量，重算时可以复用，避免重复传输和同步。

这不是主动激活卸载机制，也不要求因这条规则而保留源 CUDA 张量。条件字符串明确写的是 `cuda`，NPU 到 CPU 不会命中此分支。

### 5.2 收集 Router gate 的权重形状，优先重算匹配项

配置默认是 `force_recompute_mm_shapes_by_fqns=["moe.router.gate"]`。[形状收集代码][shape-policy]遍历当前 Block 的 `named_modules()`，通过字符串包含关系匹配完整模块路径；命中模块必须是 `nn.Linear`。这不是通用正则或 glob 匹配。

Router gate 是把每个 token 的 hidden states 投影成专家原始分数的线性层；sigmoid/softmax、可选校正偏置和 top-k 在后续路由逻辑中执行，见 [`TokenChoiceTopKRouter.forward()`][router]。gate 与 top-k 是不同算子，可以重算 gate、保留 top-k 结果。

权重形状统一为 `(D_in, D_out)`：

| 操作 | 输入及权重 | 输出 |
| --- | --- | --- |
| `mm(X, W)` | `X[T,D_in]`、`W[D_in,D_out]` | `[T,D_out]` |
| `linear(X, W)` | `X[T,D_in]`、`W[D_out,D_in]`，计算使用 `W.T` | `[T,D_out]` |

例如权重实际为 `[4,8]` 的 router linear，会将 `(8,4)` 加入 `mm_recompute_shapes`。`torch.Size((shape[1], shape[0]))` 只转换形状描述，不转置实际权重或搬运数据。

运行时 `mm/linear` 的统一形状命中集合，就返回 `PREFER_RECOMPUTE`，并跳过后面的计数。**名称只用于收集形状，运行时按形状匹配，所以其他同形状矩阵乘法也可能被选中。** 未匹配到模块时集合为空。

### 5.3 其余 mm/linear 交替保存与重算

`mm_ops` 仅包含 `aten.mm.default` 和 `aten.linear.default`。二者共享计数；特殊形状匹配项不计数。`bmm`、`addmm`、grouped matmul 不属于这个二元素元组，不参与交替计数，但仍受后面的保存集合规则管理。

```python
mode = "recompute" if ctx.is_recompute else "forward"
mm_count_key = f"{mode}_mm_count"
```

代码为正常前向和反向期间的前向重算分别计数。在本机核对的 [PyTorch SAC 实现][pt-sac]中，策略分别在 `is_recompute=False/True` 的上下文被调用：第一次决定缓存哪些输出，第二次决定取缓存还是执行算子。每次创建选择性上下文都会新建计数字典，不是所有层或整个训练过程共享的全局计数器。

这里的 `recompute` 是恢复前向激活，不是执行参数梯度算子。外部 PyTorch 可能改变策略回调协议，不能将这段双计数逻辑视为所有版本恒定的 API 行为。

在保存集合中的普通 `mm/linear`：第 1 次保存，第 2 次倾向重算，第 3 次保存，依此类推。这个规则不读取实时耗时、不比较矩阵尺寸，也不保证节省一半显存。它是计算和激活内存之间的经验折中。

### 5.4 完整优先级

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | CUDA 到 CPU 的指定拷贝 | `MUST_SAVE` |
| 2 | `mm/linear` 命中特殊权重形状 | `PREFER_RECOMPUTE`，提前返回 |
| 3 | 其余 `mm/linear` | 当前阶段计数加 1，继续判断 |
| 4 | 在 `save_ops` 中，且是偶数次 `mm/linear` | `PREFER_RECOMPUTE` |
| 5 | 在 `save_ops` 中的其余情况 | `MUST_SAVE` |
| 6 | 其他情况 | `PREFER_RECOMPUTE` |

普通逐元素加法、乘法、SiLU 等通常进入最后一条；融合后判断的是实际执行的融合算子，不能按 Python 模块名称直接推断。由于本分支关闭 early stop，重放完整区域时也可能执行某些不再需要其输出的便宜操作，不能把“优先重算”理解成最小计算量保证。

## 6. save_ops 的组成与保存理由

[`_get_default_save_ops()`][save-ops]合并三个来源：PyTorch 默认计算密集型集合、`compute_ops`、`comm_ops`。**`compute_ops` 也是需要保存输出的候选集合，不是需要重算的列表。**

### 6.1 默认集合与额外计算算子

默认集合来自 `get_default_op_list().compute_intensive_ops`。本机 PyTorch 2.7.1 的 [固定版本定义][pt-default-ops]包括 `mm`、`bmm`、`addmm`、`_scaled_mm`、卷积前后向、若干 Flash/Efficient Attention 及双线性上采样；训练环境的实际列表必须按其 PyTorch 版本检查。

仓库额外添加：

| 算子（省略部分命名空间） | 保存理由 |
| --- | --- |
| `_scaled_dot_product_cudnn_attention.default` | 避免重新执行昂贵注意力计算 |
| `_scaled_dot_product_attention_math.default` | 同上，覆盖数学实现路径 |
| `_scaled_dot_product_fused_attention_overrideable.default` | 覆盖后端可替换的融合注意力路径 |
| `flex_attention` | 保存高阶注意力算子的返回结果 |
| `linear.default` | 矩阵乘法成本通常较高，但仍受特殊形状和奇偶规则覆盖 |
| `max.default` | 源码说明用于保留量化缩放所需的最大值；规则本身不检查调用是否真的用于量化 |
| `topk.default` | 保留选中值与索引，避免重算时路由选择不一致 |
| `inductor_compiled_code` | 可用时保存编译区域输出，避免整段重做 |
| `torch_attn._varlen_attn.default` | 可用时保存自定义变长注意力输出 |

### 6.2 通信算子

`comm_ops` 显式加入 `_c10d_functional.reduce_scatter_tensor.default` 和 `_c10d_functional.all_to_all_single.default`；可选加入 `deepep.dispatch/combine.default`、`hybridep.dispatch/combine.default`。目的是复用通信输出，避免重放相同通信。

这个集合不是所有通信算子的名单，不能因此断言 AC 不会产生额外 all-gather，也不能认为它会缓存所有反向梯度同步。它管理的是 checkpoint 区域内策略实际看到的操作。

### 6.3 _resolve_ops 只是解析，不执行算子

直接的算子对象会加入结果字典；`(torch.ops, "deepep.dispatch.default")` 这样的元组则通过逐层 `getattr` 解析，缺少属性时跳过。元组形式使可选依赖未注册时也能构建集合。

返回值是 `{算子对象: MUST_SAVE}` 字典，但 `save_ops` 是集合，`set.update(dict)` 只加入字典键。最终是否保存仍由策略函数决定，字典值不会阻止后面的 mm 例外规则。

NPU `npu_grouped_matmul` 等专用算子未被这份显式补充列表列入。自定义融合算子能否被 SAC 正确缓存和重放，还取决于其注册方式与后端支持，不能只凭“也是矩阵乘法”推断行为。

## 7. 一个前向、重算和反向的具体例子

以下仅演示普通 SAC 的调用计数，不代表 Qwen3 的真实投影执行顺序。设单卡，DP=TP=CP=PP=EP=1；当前一个 microbatch 含 B=2 条序列，每条 S=3 个 token，展平后 T=6；没有梯度累积。进入 Block 时计数为 0，矩阵乘法均以可见的 `aten.mm` 执行，且都在默认保存集合中。

```text
X[6,8]
  → mm(X, W1[8,16]) = A[6,16]
  → silu(A) = B[6,16]
  → mm(B, W2[16,8]) = C[6,8]
  → C + X = Y[6,8]
```

| 操作 | 正常前向 | SAC 重算阶段 |
| --- | --- | --- |
| 第 1 次 mm | 计算并缓存 A | 复用 A，跳过该矩阵乘法 |
| SiLU | 计算 B，不由 SAC 缓存 | 从 A 重新生成 B |
| 第 2 次 mm | 计算 C，策略倾向重算 | 再次执行矩阵乘法 |
| 残差加法 | 计算 Y，策略倾向重算 | 随重放执行加法 |

FullAC 不安装上面的按算子输出缓存，重放时两次矩阵乘法和激活函数都会执行；SAC 用保留 A 的内存换取少做第一次矩阵乘法。实际峰值还受 Block 边界输入、autograd 保存、其他引用、参数聚合和分配器影响，不能只按表中一项张量计算总节省量。

第一次矩阵乘法虽然可以复用前向输出，其参数 W1 仍然可训练。checkpoint 的恢复机制让原反向继续计算 dX、dW1、dW2；“不重复做前向矩阵乘法”不等于“不计算梯度”或“冻结参数”。

普通训练在 [前向与反向函数][forward-backward]中调用 `loss.backward()`，在 [训练步后段][optimizer-step]做梯度裁剪与 optimizer step。多卡时梯度同步由既有并行机制管理，AC 不自行创建一套新的同步组。

### MoE 的状态更新还需要单独核查

除了可训练参数，前向可能更新计数 buffer。当前 [MoE 负载均衡 hook][moe-hook]通过 Block 是否带 `NO_REENTRANT` checkpoint 标记判断重算，并基于“专家计数发生两次”的假设除以 2。源码已将其标为 workaround。改变包装范围、重放次数或改为图级重算时，不能假设这个修正仍然正确；保存 top-k 也不等于所有计数副作用只执行一次。

## 8. MemoryBudgetAC：预算由编译器解释

[`MemoryBudgetAC.apply()`][budget-ac]覆盖父类 `apply()`，不遍历 Block 调用 `_wrap_block()`。其核心是设置：

```python
torch._functorch.config.activation_memory_budget = config.memory_budget
```

预算 0 对应偏向完整编译区域 AC 的激活内存端点，1 对应默认运行时间优化策略，0.5 位于两者之间。**0.5 不表示只能使用物理显存的 50%，也不保证重算一半算子。** 具体划分依赖编译器图与代价模型，不等于默认策略必然运行逐算子硬件 profiling。

可选 `visualize_memory_budget_pareto` 会设置输出目录和可视化相关编译器配置。`memory_budget` 在配置阶段要求位于 `[0,1]`；Trainer 和 Forge 还检查模型编译是否开启。本分支通过进程级 functorch 配置传递预算，部署多模型或复用进程时应留意其作用范围。

## 9. GraphTrainer 的图级重计算与卸载

GraphTrainer 的 `aot_fx_trace` 路径先捕获前向、loss 和反向图，再运行内存变换，见 [GraphTrainer 入口][graph-trainer]和 [pass 顺序][graph-passes]：

```text
tag_with_memory_policy_pass
    → 为图节点标记保存、重算或 CPU 卸载
apply_cpu_offload_pass
    → 为卸载节点插入搬运与等待
selective_activation_remat_pass
    → 在反向消费者之前插入重算副本，改写依赖
```

| `compile.memory_policy` | 实际含义 |
| --- | --- |
| `default` | 保存计算密集型算子，结合 FSDP/层边界约束；不使用普通 mm 奇偶启发式 |
| `eager` | 使用按层重置的 mm/linear 交替规则；这个名字不表示关闭重算 |
| `full` | 尽量重算，保留必要边界、随机算子以及显式指定的保存项 |
| `sac_and_offload` | 先执行 default SAC，再在预算内卸载剩余符合条件的激活 |

这些策略见 [内存策略实现][graph-memory]。`eager` 图策略模拟奇偶规则，但其函数没有普通 SAC 的 router 权重形状特判或 CUDA 到 CPU 特判，不能认为两条实现逐条完全相同。

`full` 策略支持通过 `full_recompute_save_ops` 指定 `MODULE_FQN_PATTERN::OP`，多个项用 `|` 分隔。这是在完整重算中指定保存例外；与普通 SAC 的字符串包含匹配及形状集合不同。配置和校验见 [GraphTrainer 配置][graph-config]及 [selector 解析][graph-selectors]。

[重算 pass][graph-remat]会复制所需前向节点，在反向前恢复依赖；对于仍有前向消费者的原节点保留原计算，仅删除没有用户的原节点。该路径对 RNG 重放和多个独立反向重算区域有限制，full 策略专门保存随机算子；不能将它与 eager FullAC 视为完全等价。

卸载使用 `MUST_CPU_OFFLOAD` 标记和上游私有 `ao.offload/reload/wait_tensor`，由 [专用 pass][graph-offload]真正插入搬运。枚举成员存在不等于普通 SAC 自动支持卸载；在普通策略里改一个返回值，不能替代完整异步搬运实现。

这条实验路径不是普通 K3 Trainer 自动执行的步骤。GraphTrainer 的模型入口、融合算子图捕获、并行包装与设备搬运后端都要分别适配；本文不声明 K3 NPU 已可使用图级 SAC 或异步卸载。

## 10. K3 NPU 边界与验证建议

本次分支的 [K3 `parallelize_kimi_k3()`][k3-parallelize]保留普通 AC 接入口，并会对 decoder 和存在的视觉编码器分别应用策略。但是 [debug 配方][k3-config]设置 `activation_checkpoint=None`：注释记录目标 NPU 环境中 selective checkpoint 与 `npu_grouped_matmul` 等自定义算子存在兼容性问题，可能报 `aten.lift_fresh ... not found in storage`。这是源码记录，本文未在 NPU 复现，也不据此宣称所有 torch_npu 版本均不支持 SAC。

同一并行化函数还禁止模型编译，并拒绝 TP/PP/CP/EP，因此当前分支不能直接通过开启 `MemoryBudgetAC` 解决问题，也不能将另一个支持 EP 的分支结论套用到这里。

后续验证应至少覆盖：

1. 固定模型、输入和并行配置，对比关闭 AC、FullAC、SAC 的 loss、梯度范数及关键参数梯度；按仓库要求使用 seed=42 和确定性设置，不用 `deterministic_warn_only` 放过问题。
2. 确认融合 GMM、注意力和 KDA 在前向与重算中的行为，检查可选算子注册以及保存集合实际命中情况。
3. 检查 MoE 路由选择、专家计数和 bias 更新；检查分块 loss 的 lm_head 梯度，不能只看 loss 接近。
4. 再扩展多卡与多模态，观察 FSDP 参数聚合、梯度同步和重算之间的交互。
5. 性能比较至少运行 10 步，分开观察启动与稳定阶段，记录峰值内存、步耗时和吞吐；本文没有给出实测收益。

GraphTrainer 已有 [SAC 峰值内存与数值测试文件][graph-tests]，这里只确认测试代码存在，未执行这些测试，也未运行本分支训练。文档检查不等于训练功能验收。

## 11. 继续阅读顺序

理解普通 AC，建议沿以下路径阅读：

1. [模型配置选择][qwen-recipe] → [ModelSpec 绑定函数][qwen-spec] → [Trainer 传参][trainer-parallelize]。
2. [模型并行化接入口][qwen-parallelize] → [配置 build][config-build] → [逐层 apply][ac-apply]。
3. [FullAC 包装][full-ac] → [SAC 策略][sac-policy] → [保存集合][save-ops]。
4. [训练前后向][forward-backward] → [优化器更新][optimizer-step] → [MoE 状态修正][moe-hook]。
5. 再比较 [预算策略][budget-ac] 与 [GraphTrainer 图级重算][graph-remat]，避免混用三条路径的开关与假设。

[ac-types]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L336
[trainer-budget-check]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/trainer.py#L163
[trainer-default]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/trainer.py#L112
[qwen-recipe]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/qwen3/config_registry.py#L37
[k3-config]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/kimi_k3/config_registry.py#L99
[manager]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/config/manager.py#L34
[qwen-spec]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/qwen3/__init__.py#L613
[model-spec]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/protocols/model_spec.py#L34
[entry]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/train.py#L17
[trainer-parallelize]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/trainer.py#L483
[qwen-parallelize]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/qwen3/parallelize.py#L27
[config-build]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/config/configurable.py#L134
[ac-apply]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L152
[full-ac]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L173
[selective-ac]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L214
[forward-backward]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/trainer.py#L782
[forge-config]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/forge/engine.py#L54
[forge-parallelize]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/forge/engine.py#L256
[sac-policy]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L246
[shape-policy]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L220
[router]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/common/moe.py#L277
[save-ops]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L31
[optimizer-step]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/trainer.py#L923
[moe-hook]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/components/optimizer/optimizer.py#L443
[budget-ac]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/distributed/activation_checkpoint.py#L290
[graph-trainer]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/trainer.py#L198
[graph-passes]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/passes.py#L255
[graph-memory]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/memory_policy.py#L345
[graph-config]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/configs.py#L97
[graph-selectors]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/memory_policy.py#L97
[graph-remat]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/selective_activation_remat.py#L101
[graph-offload]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/cpu_offload.py#L412
[k3-parallelize]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/models/kimi_k3/parallelize.py#L24
[graph-tests]: https://github.com/the-fall-moon/torchtitan/blob/a8c8a0fcef72396652d2da56fadbb27ff1baa8ec/torchtitan/experiments/graph_trainer/tests/test_sac_peak_memory.py#L87
[pt-wrapper]: https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py#L114
[pt-sac]: https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/utils/checkpoint.py#L1269
[pt-default-ops]: https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/_functorch/partitioners.py#L1514
