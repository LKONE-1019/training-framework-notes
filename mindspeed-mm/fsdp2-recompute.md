# FSDP2 后端重计算：功能、配置与实现链路

[返回 MindSpeed-MM 索引](README.md) · [返回仓库首页](../README.md)

- 上游：[Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM)；源码基线：`dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc`。
- 更新日期：2026-10-09。范围：`mindspeed_mm/fsdp/` 插件式 FSDP2，以 Qwen3.5 35B 配置为例。
- 验证：源码静态分析，未运行训练或分布式数值验证；源码链接的路径与行号已按本地固定版本核对。

## 1. 功能与整体流程

重计算通过减少长期保存的中间激活来节省设备内存，反向需要这些激活时再执行对应计算。当前 MM 按模块路径或方法路径选择重计算区域，把目标函数包装成 PyTorch `checkpoint()` 调用。

这里的 checkpoint 是“激活检查点”：保留重算所需信息，与保存权重到磁盘的模型 checkpoint 不同。它管理激活的保存与恢复；FSDP 则负责参数、梯度的分片和通信。

```text
YAML：features.recompute / recompute_plan
  → ConfigManager → args.features
  → Trainer.get_model()：创建模型
  → FeaturesApplier.pre_fully_shard_apply()
      → apply_recompute_models()：检查开关
      → recompute_modules()：匹配目标、替换 forward 或方法
  → ParallelApplier：应用并行与 FSDP 分片
  → model(...)：进入包装后的函数 → checkpoint
  → loss.backward()：checkpoint 按需重算 → 继续反向
```

安装包装和实际重算发生在不同阶段：前者在构建模型时，后者在训练反向中。

## 2. 参数：选择重计算区域与执行方式

[RecomputePlanConfig][plan] 定义计划字段，[apply_recompute_models()][enable] 检查开关。以下配置片段选中视觉 blocks 和语言 decoder layers：

```yaml
features:
  recompute: true
  recompute_plan:
    apply_modules:
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
    use_reentrant: false
```

| 参数 | 默认/未配置行为 | 作用 |
| --- | --- | --- |
| `features.recompute` | 入口按 `False` 处理 | 总开关 |
| `recompute_plan.apply_modules` | `[]` | 模块或方法路径；空列表不安装目标 |
| `recompute_plan.use_reentrant` | `false` | 选择 checkpoint 实现 |
| `recompute_plan.flatten_inputs` | `false` | 非重入模式下，展平参数树后交给 checkpoint |
| `recompute_plan.op_replay_scopes` | `[]` | 指定算子回放区域与白名单 |

### 模块路径与方法路径

路径相对于传给 `recompute_modules()` 的模型根节点，以 `model.named_modules()` 为准。

| 示例 | 含义 |
| --- | --- |
| `model.language_model.layers.3` | 第 3 层的 `forward` |
| `model.language_model.layers.{*}` | 数字索引对应的所有层 |
| `model.language_model.layers.{0-3}` | 第 0～3 层，闭区间 |
| `model.language_model.layers.{*}._attention_forward` | 匹配层上的指定方法，前提是方法存在且会被调用 |

[module_name_match()][match] 使用扩展通配符，不直接接受任意正则。普通 `*` 可以跨越路径中的点，范围可能包含子模块；`{*}` 只匹配数字。配置相互重叠时，[get_recompute_modules()][matchmodules] 不会去重，可能重复包装同一个实例。

## 3. 从配置到函数包装的调用链

| 顺序 | 代码入口 | 完成的工作 |
| --- | --- | --- |
| 1 | [训练入口][entry] → [ConfigManager.load_and_parse()][config] | 读取并合并配置，生成含 `features` 的参数对象 |
| 2 | [Trainer.get_model()][trainer] | 创建基础模型；启用 LoRA 时先安装 adapter |
| 3 | [pre_fully_shard_apply()][features] | 在 FSDP 分片前安装重计算等特性 |
| 4 | [apply_recompute_models()][enable] | 检查总开关和计划；需要 Op Replay 时取得缓存 |
| 5 | [recompute_modules()][recompute] | 将目标分成模块级和方法级，收集实例并安装包装 |
| 6 | [ParallelApplier.__call__()][parallel] | 应用量化、并行、FSDP 和预取配置 |
| 7 | [TrainEngine 前向][engine]、[反向][backward] | 正常调用模型和 `loss.backward()`，由 checkpoint 接管重算 |

`recompute_modules()` 先判断完整 pattern 是否匹配模块。匹配成功就包装 `module.forward`；否则按最后一个点拆成“模块 pattern + 方法名”，由 [_apply_recompute_method()][method] 处理。

`get_recompute_modules()` 返回的是 `(名称, 模块对象)` 列表，实际安装发生在后续赋值：

```python
module.forward = recompute_wrapper(
    module.forward, plan.use_reentrant, context_fn,
    plan.flatten_inputs, module
)
```

这是原地替换实例方法。配置路径不存在、指定方法不存在等情况会在匹配和解析阶段报错。

特性安装的完整顺序为：准备 Chunk Layer → 重计算 → 激活卸载 → 完成 Chunk Layer 外层调度 → Chunk MBS → chunkloss。后安装的函数包装位于外层；未启用的功能跳过。重计算在 `ParallelApplier` 调用前就已安装，阅读调用顺序应以实际执行语句为准。

## 4. recompute_wrapper 与 checkpoint 做了什么

[recompute_wrapper()][wrapper] 保留原始函数，返回新的 wrapper；后续模块被调用时才执行它。忽略附加策略，等价于：

```python
# 原来
output = original_forward(x)
# 包装后
output = checkpoint(original_forward, x, use_reentrant=False)
```

包装函数有三个主要动作：

1. 安装时读取函数签名；实际调用时，若签名显式包含 `past_key_values`，把该参数槽位设为 `None`。位置传参和关键字传参都能处理。
2. 非重入模式下，组合可选的 Op Replay、MXFP8 上下文；需要时展平输入，然后调用 checkpoint。
3. 重入模式下，用 `functools.partial` 固定 kwargs，再将位置参数交给 checkpoint。

标准训练入口还会向模型传入 `use_cache=False`。重计算使用当前 batch 的已有输入，不重新读取数据，也不额外执行 optimizer step。

### use_reentrant 的区别

| 行为 | `true`：重入 | `false`：非重入，MM 默认 |
| --- | --- | --- |
| 首次前向 | 在 `no_grad` 下执行区域内部计算 | 记录 autograd 图，但不按普通方式长期保存所有内部激活 |
| 反向重算 | 完整重跑函数，建立图后执行内部反向 | 缺少激活时重算，默认可在恢复所需信息后提前停止 |
| `context_fn` | 不支持 | 支持，用于 Op Replay 等 |
| `flatten_inputs` | 当前 MM 分支不处理 | 可启用 |

记录计算图与保留全部激活张量是不同的事。checkpoint 的默认 RNG 保存有助于 dropout 等重算对齐，函数的输入、状态和副作用仍需可重放。详细语义见 [PyTorch 2.10 checkpoint 文档](https://docs.pytorch.org/docs/2.10/checkpoint.html)。

### flatten_inputs 的用途

[_flatten_call()][flatten] 展开整个 `(args, kwargs)` 参数树，按对象身份去重，再把叶子作为位置参数交给 checkpoint；调用原函数前恢复结构。

```text
forward(x, extra={"mask": mask}, residual=x)
  → checkpoint 接收唯一叶子 x、mask 等
  → 恢复原调用，residual 和第一个参数仍指向同一对象
```

它不改变张量维度。主要用途是让 kwargs、容器里的边界张量也经过 checkpoint 的保存通道，便于 ActStash 管理。

## 5. 与激活卸载配合：哪些保存，哪些重算

以一个普通 MLP 为例，假设它同时被非重入 checkpoint 和 ActStash 包装，输入 `x` 通过位置参数传入，未启用算子回放：

```text
x → Linear1 → a → GELU → b → Linear2 → y
```

| 对象 | 处理方式 |
| --- | --- |
| 输入 `x` | 作为重算边界输入保存，符合条件时交给 ActStash 缓存与卸载 |
| 中间结果 `a、b` | 由 checkpoint 管理，反向需要时重算恢复 |
| 输出 `y` | 继续交给下一层；其保存和卸载取决于后续计算与包装范围 |
| 权重 | 由模型和 FSDP 管理，不是普通激活卸载的目标 |

输入 `x` 也是激活：它可能就是上一层的输出。两种特性可以组合，因为重计算减少内部激活的保存，卸载管理剩余需要保存的激活。

```text
前向：保存 x → 执行 MLP → 得到 y
反向：取回 x → 重算所需的 a、b → 求参数梯度和输入梯度
```

判断过程来自嵌套保存 hook，而非框架根据变量名自动分类。算子的 autograd 实现提出保存需求；checkpoint 接管区域内部的保存；外层 ActStash 捕获边界输入的保存，并筛选适合缓存的设备张量。

两种卸载实现的范围不同：

- [legacy 包装][legacy]（默认）：识别位置参数中的 `hidden_states`，筛选与其 `data_ptr()` 相同的合格待保存张量；同时维护层索引，供专用 skip-recompute 算子使用。
- [ActStash][stash]（`impl: stash`）：使用 `saved_tensors_hooks` 与共享 SwapCache；包在 checkpoint 外时主要覆盖边界输入，配合 `flatten_inputs` 可覆盖嵌套参数。设备内存是否真正释放还取决于换出策略与外部引用。

## 6. Qwen3.5 35B：实际配置下的一次前向与反向

本节对应 [qwen3_5_35B_config.yaml][qwenconfig]，模型实现为 `qwen3_5_moe`，在 NPU 上使用 AscendC GDN。关键配置如下：

| 项目 | 该配置的取值 |
| --- | --- |
| 重计算范围 | `model.visual.blocks.{*}`、`model.language_model.layers.{*}` |
| checkpoint 类型 | 未显式设置，默认非重入 |
| 激活卸载 | 开启，同样覆盖视觉 blocks 和语言 layers；默认 legacy |
| 专用跳过重算 | `skip_gdn_recompute: true`、`skip_flash_attn_recompute: true` |
| 通用 Op Replay | 未配置 `op_replay_scopes` |
| 语言层 Chunk MBS | 开启，`chunk_mbs: 2` |
| 训练 batch | `micro_batch_size: 4`，梯度累积步数 1 |

### 6.1 每个语言层按两组样本进入 checkpoint

忽略外部 FSDP 生命周期 hook，语言层的函数包装关系是：

```text
Chunk MBS → legacy 激活卸载 → checkpoint → 原始 DecoderLayer.forward
```

某层输入形状为 `[4,S,H]` 时，[Chunk MBS][chunkmbs] 先处理样本 0、1，再处理样本 2、3；各组进入独立 checkpoint 调用，输出合并回 `[4,S,H]` 后交给下一层。这是单层内部的 batch 分组，不是两次完整 LLM 训练 step。

### 6.2 原始 decoder layer 的计算结构

[Qwen3_5MoeDecoderLayer][decoder] 根据 `layer_types` 选择 GDN 或 Full Attention；同一层只执行其中一种。其主流程为：

```text
输入 x
  → Input RMSNorm
  → GDN 或 Full Attention
  → 残差相加
  → Post-Attention RMSNorm
  → MoE：Router、选中的 Experts、Shared Expert
  → 残差相加 → 输出
```

attention 入口在 [_attention_forward()][attention]，专家计算在 [SparseMoeBlock.forward()][moe]。

### 6.3 保存与复用哪些结果

这份配置同时有整层 checkpoint 和专用 `skip_*_recompute`，因此层内部也会额外保存部分昂贵计算的结果：

| 区域 | 首次前向 | checkpoint 重算经过该区域时 |
| --- | --- | --- |
| 普通归一化、投影、MoE 等 | 正常计算，内部激活由 checkpoint 管理 | 按反向需求重新执行 |
| Full Attention 核心 | 保存 `attn_output、softmax_max、softmax_sum` | 取回这些结果，跳过对应 FA 核心前向计算 |
| AscendC GDN 核心 | 保存 `g、o、A`，需要时另存 `final_state` | 取回结果，跳过对应 GDN 核心前向计算 |

代码：[Flash Attention 保存/取回][skipfa]、[AscendC GDN 保存/取回][skipgdn]。这些分支使用 legacy `OffloadManager`，并对最后一个受管理层保留设备驻留路径，不能认为所有结果都必须搬到 CPU。

以 Full Attention 层的一组输入为例：

```text
loss.backward()
  → checkpoint 需要恢复该层激活
  → 取回输入 x
  → 重算 RMSNorm、Q/K/V 投影及所需处理
  → FA 专用分支取回原先结果，跳过核心前向算子
  → 继续所需的投影、残差、归一化和 MoE 计算
  → 所需激活恢复后，继续反向求梯度
```

非重入 checkpoint 可以提前结束重算，不保证以上所有步骤每次都完整重跑。跳过核心前向重算也不表示跳过其梯度：仍执行 [FA backward][fabwd] 和 [GDN backward][gdnbwd]。

该配置的 `skip_*_recompute` 与下一节的通用 Op Replay 是两套接入方式。若改用 ActStash，应同时审查这些专用算子的缓存释放协议；[ActStash 的交互说明][stashinteraction] 指出，它不维护 legacy 的层索引/释放协议，可能降低相关张量的显存释放收益。

## 7. Op Replay 与 FSDP 的两个衔接点

### 7.1 op_replay_scopes 与 op_cache

`op_replay_scopes` 是规则：指定在哪些模块区域，缓存哪些底层算子的结果；`op_cache` 是运行时保存这些结果的缓存对象。

[OpReplayScopeConfig][scope] 包含 `apply_modules`、`cache_ops`、`save_rng` 和日志标签 `name`。例如在 checkpoint 覆盖的 MLP 区域，只缓存实际执行到的 `aten.mm.default` 输出。缓存匹配的是底层算子，不是 Python 层的类名。

[SwapManager.get_cache("op_replay")][cache] 返回共享 SwapCache，负责结果的设备/CPU 存储与取回。Op Replay 为每次 checkpoint 建立独立的调用记录，将算子结果与句柄对应起来：

```text
首次前向：执行命中算子 → op_cache.put(输出) → 记录句柄
重算：按本 checkpoint 的调用记录取回输出 → 跳过算子
未命中：回退到实际重算
```

它通过 [context_fn][replay] 区分首次执行与重算，要求 `use_reentrant: false`，且 scope 处于实际执行的 checkpoint 区域内。启用后，[嵌套检查][nested] 拒绝同时包装父模块和子孙模块；方法级调用关系无法只靠模块树判断，因此仅警告，需检查是否形成嵌套。

MXFP8 的模块标记与重算上下文也接入 `recompute_wrapper`，方法级目标不应用该模块标记。入口见 [低精度重算适配][mxfp8]；其具体行为还依赖外部 `fsdp_turbo` 实现。

### 7.2 FSDP 必须为重算准备好参数

checkpoint 重算仍要使用模型参数。若一个 decoder layer 内有独立分片的 attention 或 experts，就需要让这些子单元的参数还原与重计算边界配合。

Qwen3.5 35B 配置将语言层设为 `parallel.fsdp_plan.hook_modules`。[FSDP 安装器][fsdp] 找到对应父模块后，以 `fully_shard(child, hook_module=parent, ...)` 挂载；[自定义 patch][fsdppatch] 让父 checkpoint block 的生命周期 hook 管理相关嵌套单元，支持重算时的参数还原。

| 配置 | 职责 |
| --- | --- |
| `features.recompute_plan.apply_modules` | checkpoint 的计算边界 |
| `parallel.fsdp_plan.apply_modules` | 参数分片单元 |
| `parallel.fsdp_plan.hook_modules` | FSDP 生命周期 hook 的挂载位置 |

三者不是自动等同。该版本 FSDP patch 明确适配 PyTorch 2.7.1、2.9.0、2.10.0；重计算沿用正常的 `loss.backward()`、梯度同步和优化器更新流程。

仓库的 [重计算接线测试][tests] 覆盖模块/方法包装、反向重放和 KV 参数处理。本文未执行这些测试或上述 35B 配置，数值正确性及显存/吞吐收益仍需在目标设备上验证。

[plan]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/params/feature_args.py#L86
[enable]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/apply_features.py#L63
[match]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/utils/str_match.py#L4
[matchmodules]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L169
[entry]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/train/trainer.py#L554
[config]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/config/config_manager.py#L88
[trainer]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/train/trainer.py#L214
[features]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/apply_features.py#L213
[recompute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L69
[parallel]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/distributed/torch_parallelize.py#L77
[engine]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/train/train_engine.py#L209
[backward]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/train/train_engine.py#L242
[method]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L119
[wrapper]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L215
[flatten]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L180
[legacy]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L331
[stash]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/act_stash.py#L71
[qwenconfig]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/examples/qwen3_5/qwen3_5_35B_config.yaml#L95
[chunkmbs]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/communication/chunk_mbs.py#L113
[decoder]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1511
[attention]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1595
[moe]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1466
[skipfa]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/flash_attn/skip_recompute_flash_attn.py#L14
[skipgdn]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/gdn/flash_gated_delta_rule.py#L436
[fabwd]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/flash_attn/skip_recompute_flash_attn.py#L120
[gdnbwd]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/gdn/flash_gated_delta_rule.py#L514
[stashinteraction]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/act_stash.py#L28
[scope]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/params/feature_args.py#L57
[cache]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/swap_manager.py#L40
[replay]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/op_replay.py#L397
[nested]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L146
[mxfp8]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/recompute.py#L19
[fsdp]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/distributed/fully_shard_parallel.py#L89
[fsdppatch]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/fully_shard/fully_shard.py#L875
[tests]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/tests/ut_fsdp/features/memory/test_recompute_functions.py#L1
