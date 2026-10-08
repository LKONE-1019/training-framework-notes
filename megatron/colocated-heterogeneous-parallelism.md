# Megatron 共卡异构并行：从 TP/DP 布局转换到梯度同步

本文分析视觉 encoder 与语言模型共用同一批 GPU、采用不同 TP/DP 配置时，Megatron-Core MIMO 如何组织通信、转换特征布局、恢复反向梯度，并完成参数更新。这里的“异构”指**模块之间的并行布局不同**，不指不同型号 GPU。

| 项目 | 范围 |
| --- | --- |
| 外层仓库 | [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) |
| Bridge commit | `a393057e71aa7289f34b67138352070de333d3e2` |
| 核心实现 | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)，当前 Bridge 的 `3rdparty/Megatron-LM` 子模块 |
| Megatron-Core commit | `07147d6942dfd4e6d1566b64bb1730f478a1bc32`，与 Bridge gitlink 一致 |
| 分析日期 | 2026-10-08 |
| 适用范围 | MIMO 显式 grid 的共卡 TP/DP 转换；PP=CP=1，示例 EP/ETP/GTP 均为 1；CUDA/NCCL |
| 验证状态 | 本地源码静态分析及文档检查，未执行 GPU 分布式训练或数值对齐测试 |

两个源码仓库的受版本控制文件均无本地修改。分析以固定提交源码为准。下文源码链接已核对本地文件、符号与行号；远端页面读取失败，未验证所有源码链接的远端可访问性。

## 1. 核心结论与接入边界

共卡异构的主线是：**同一物理 rank 在不同模块中有不同 TP/DP 坐标，encoder 输出经过 batch 维转换后，才进入 LLM 的 embedding 合并。** 每个模块按自己的 TP 切分权重，按自己的 DP 同步参数梯度。

需要分清当前仓库的两个层次：

| 层次 | 当前源码行为 |
| --- | --- |
| Megatron-Core `MimoModel` / `ColocatedBridgeCommunicator` | 有共卡分支、自定义 autograd 和对应正确性测试 |
| Megatron-Core `examples/mimo/training/topology.py` | 拓扑校验允许所有模块完整共用 ranks，或互不重叠地覆盖 world |
| Bridge `MegatronMIMOParallelismConfig` | `finalize()` 调用 `_validate_heterogeneous()`，拒绝模块 rank 区间重叠 |
| Bridge Qwen3.5-VL MIMO 示例 | README 将共卡训练标为 Planned，当前示例说明的是分卡训练 |

因此，本文梳理的是**当前 Bridge 所固定的 Core 实现及其训练辅助代码**，不能据此声称 Bridge 高层 recipe 已能直接配置任意共卡布局。把 Bridge 中两个模块的 `rank_offset` 都设为 0，会先遇到重叠校验。[Core 拓扑校验][topology-validate]、[Bridge 配置校验][bridge-config]、[Qwen3.5-VL 示例状态][bridge-example]。

另一个容易混淆的功能是 Nemotron Omni 的 `vision_dp_over_cp`：它借用 LLM CP 组分摊图像编码，属于另一条执行路径。本文的 `ColocatedBridgeCommunicator` 明确要求 CP=1，不能把两者的约束或效果合并。[共卡校验][comm-validate]、[Omni 参数][omni-provider]。

## 2. 整体调用流程

```text
为 encoder、language 分别创建 HyperCommGrid 和 ProcessGroupCollection
  → 准备各模块的 TransformerConfig 与 ModuleSpec
  → MimoModelConfig.module_to_grid_map
  → RankRole.build() 判断为 COLOCATED
  → 每个模态创建 ColocatedBridgeCommunicator
  → 按各自 spec 构建模块与权重分片
  → 各模块单独包装 DDP，接入梯度同步 hook 和 MimoOptimizer

每次 microbatch：
  输入按各模块 DP 布局准备
    → encoder + 模态侧 projector
    → _apply_colocated_comms()
        FAN_IN：all-gather / FAN_OUT：narrow / EQUAL：透传
    → 按特殊 token 位置合并视觉与文本 embedding
    → LLM forward → loss → autograd backward
    → bridge backward 恢复 encoder 所需 activation gradient

所有 microbatch 结束：
  → 计算全局有效 token 数 N_global
  → 各模块分别完成参数梯度归约
  → 各模块统一乘 1 / N_global
  → MimoOptimizer.step()
```

这里有三种不同的通信职责，不能互相替代：

- **模块内部 TP 通信**：配合该模块的权重切分完成计算。
- **bridge 通信**：对齐 encoder 与 LLM 之间的 activation / activation gradient 布局。
- **模块内部 DP 通信**：归约同一参数分片在不同样本上的梯度。

共卡表示共享设备和 rank 集合；上述前向代码在每个 rank 上依次执行 encoder 和 LLM，不表示两个模块自动同时运行，也不自动提供跨模块流水线重叠。

## 3. 配置如何变成实际模块和通信组

### 3.1 MimoModelConfig 负责连接模块

关键字段见 [`MimoModelConfig`][model-config]：

| 字段 | 默认值 / 含义 | 共卡路径中的作用 |
| --- | --- | --- |
| `language_model_spec` | ModuleSpec | LLM 的构造规格，包含实际 TransformerConfig |
| `modality_submodules_spec` | 空字典 | 如 `images` 对应的 encoder 及模态侧 projection |
| `module_to_grid_map` | `None` | 指定模块使用哪个 grid；语言模块固定使用键 `language` |
| `special_token_ids` | 空字典 | 定位视觉等模态特征写入 token 序列的位置 |
| `language_model_input_projections_spec` | 空字典 | 当前非空时要求 NON_COLOCATED，共卡不能启用语言侧 projection |
| `kv_format` | `"sbhd"` | 传给序列分区适配器，不决定 bridge 方向 |

`__post_init__()`校验 projection 名称和 grid map 的键，**不会把 grid 的 TP/DP 自动写入任意 ModuleSpec**。构建方必须让 TransformerConfig、ProcessGroupCollection 和 grid 一致。

[测试构建器 `get_mimo_model()`][model-builder]展示了实际接线：分别获得 `vision_pg`、`language_pg`，传入各模块的 spec，创建 `MimoModel`，再用各自 PGC 包装 DDP。共卡仍然有两个独立的并行配置；不需要在前向中反复改写一套全局 `parallel_state` 来切换模块。

尤其要注意：[语言侧 projection 初始化][language-projection]会对共卡非空配置抛错。本文示例使用模态子模块内部的 projector，使输出隐藏维度在进入 bridge 前已与 LLM 对齐。

### 3.2 HyperCommGrid 描述同一批 rank 的不同坐标

[`HyperCommGrid`][grid]将 `shape`、`dim_names`、`rank_offset` 映射成 rank 枚举。grid 描述布局，`create_pg()`才创建通信组，`get_pg()`取得当前 rank 所属组。

下面是 8 卡布局声明示意，不是完整训练脚本：

```python
encoder_grid = HyperCommGrid(
    shape=[2, 1, 4, 1],
    dim_names=["tp", "cp", "dp", "pp"],
    rank_offset=0,
    backend="nccl",
)
language_grid = HyperCommGrid(
    shape=[4, 1, 2, 1],
    dim_names=["tp", "cp", "dp", "pp"],
    rank_offset=0,
    backend="nccl",
)
module_to_grid_map = {
    "images": encoder_grid,
    "language": language_grid,
}
```

TP 是变化最快的坐标，两张 grid 都覆盖 ranks 0～7：

| 物理 ranks | encoder：TP=2、DP=4 | LLM：TP=4、DP=2 |
| --- | --- | --- |
| 0、1 | DP0，TP0/1 | DP0，TP0/1 |
| 2、3 | DP1，TP0/1 | DP0，TP2/3 |
| 4、5 | DP2，TP0/1 | DP1，TP0/1 |
| 6、7 | DP3，TP0/1 | DP1，TP2/3 |

例如 rank5 在源 grid 中是 `(dp=2,tp=1)`，在目标 grid 中是 `(dp=1,tp=1)`。同一进程参与两套组，但这些坐标不会改变它所绑定的物理设备。

若要连到训练循环，还必须创建 TP、DP 等实际组并组装 PGC；[Core `create_topology()` / `_build_grid()`][topology-build]提供了完整构造路径。仅构造上面两个 grid 还不构成可训练模型。

### 3.3 RankRole 选择前向路径

[`RankRole.build()`][role]在未提供 grid map 时保留旧的 COLOCATED 路径；显式提供 map 时，比较各 grid 的 `rank_offset` 和 `size`：

- 全部相同：COLOCATED，每个 rank 拥有所有模态与语言模块。
- 不全相同：进入 NON_COLOCATED 角色构建。

这里没有计算 FAN_IN/FAN_OUT；方向由后面的 communicator 决定。角色判断也不能代替完整拓扑校验：Core training helper 额外拒绝部分重叠或不能覆盖 world 的布局。

[`MimoModel.__init__()`][model-init]只在“COLOCATED 且有显式 grid map”时创建共卡通信器。只要任一模块的 grid 缺少 `tp` 或 `dp` 维，[`_build_colocated_communicators()`][model-comm]会跳过创建。因此“角色是共卡”不等于“异构重分布已接入”。

## 4. 通信器如何决定方向和 gather 组

[`ColocatedBridgeCommunicator.__init__()`][comm-init]依次校验 grid、提取 TP/DP、建立两套 rank 坐标映射，再选择方向：

| 源与目标 DP | direction | scale | forward | backward |
| --- | --- | --- | --- | --- |
| `src_dp > dest_dp` | FAN_IN | `src_dp / dest_dp` | all-gather | narrow |
| `src_dp < dest_dp` | FAN_OUT | `dest_dp / src_dp` | narrow | all-gather |
| 相等 | EQUAL | 1 | contiguous 透传 | contiguous 透传 |

构造校验要求：相同 rank span、显式 TP/DP、PP/CP 若存在则为 1、两侧 DP 可互相整除。本文示例采用标准 TP/DP 排列，不将其推广到任意维度排列或 GTP/专家维度重排。

[`_build_rank_mappings()`][comm-mapping]枚举每个 grid 的 TP 组，将外层组号作为 DP 索引，组内位置作为 TP 索引。随后 `_build_gather_groups()`：

- FAN_IN：固定目标 DP 索引及源 TP 索引，收集连续的 `scale` 个源 DP rank。
- FAN_OUT：固定源 DP 索引及目标 TP 索引，收集连续的 `scale` 个目标 DP rank。

对于上面的 TP2/DP4 → TP4/DP2：

```text
encoder TP 组：[0,1] [2,3] [4,5] [6,7]
encoder DP 组：[0,2,4,6] [1,3,5,7]
language TP 组：[0,1,2,3] [4,5,6,7]
language DP 组：[0,4] [1,5] [2,6] [3,7]

bridge gather 组：[0,2] [1,3] [4,6] [5,7]
```

bridge gather 组既不是完整 encoder DP 组，也不是 LLM TP 组。它只收集映射到同一目标 DP 副本的那几份样本。源码通过 `dist.new_subgroups_by_enumeration(..., backend="nccl")`将 rank 列表变为实际 ProcessGroup；当前 rank 对应的句柄保存在 `gather_pg`。

方向反转为 TP4/DP2 → TP2/DP4 时，此例的 gather rank 列表恰好相同，但它在反向传播时使用，不能由列表相同推导两种方向行为相同。[rank 与 group 测试][comm-tests]明确列出这两种情况。

## 5. 输入必须先按模块 DP 对齐

bridge 处理的是**encoder 输出**，不会替调用方准备 encoder 原始图像，也不会自动改写 LLM 的文本、labels 和 loss mask。

设一个逻辑 microbatch 跨 DP 共含 `Gμ` 条不同样本，均匀切分时：

```text
encoder 本地样本数 B_enc = Gμ / DP_enc
LLM 本地样本数     B_lm  = Gμ / DP_lm

Gμ = B_enc × DP_enc = B_lm × DP_lm
一次优化器更新的样本数 = Gμ × 梯度累积 microbatch 数
```

这里 TP rank 处理同一份样本的不同权重分片，不能再把 TP 乘入样本总数。

[正确性测试的 `_slice_global_batch_for_dist()`][data-slice]先把同一全局 batch 按 DP 较小的一侧切成较大的本地 batch；[`forward_step()`][forward-step]再处理另一侧：

- FAN_IN：先按 LLM DP 切分；保留文本批次，再将模态输入缩小为 encoder 对应的样本片段。
- FAN_OUT：先按 encoder DP 切分；保留模态批次，再缩小 `input_ids/labels/loss_mask/position_ids` 到 LLM 对应片段。

这是测试中的显式数据准备逻辑，不能假设任何已有 Bridge DataLoader 都会自动完成它。真实图片路径还必须保持“文本样本 → 图片 → patch/视觉 token”的顺序一致。

## 6. 8 卡 FAN_IN：四份 encoder batch 合成两份 LLM batch

取 PP=CP=1，encoder TP=2/DP=4，LLM TP=4/DP=2。一个逻辑 microbatch 共 8 条样本 A～H；每条样本经过 encoder/projector 后得到 3 个视觉 token，隐藏维度为 `H_lm`，文本含占位符的总序列长度为 16。

| ranks | encoder 本地样本 | bridge 前视觉特征 | LLM 本地样本 | bridge 后视觉特征 |
| --- | --- | --- | --- | --- |
| 0、1 | A、B | `[6,H_lm]` | A、B、C、D | `[12,H_lm]` |
| 2、3 | C、D | `[6,H_lm]` | A、B、C、D | `[12,H_lm]` |
| 4、5 | E、F | `[6,H_lm]` | E、F、G、H | `[12,H_lm]` |
| 6、7 | G、H | `[6,H_lm]` | E、F、G、H | `[12,H_lm]` |

在 `[0,2]` 组中，rank0 提供 A/B 特征，rank2 提供 C/D 特征；all-gather 后双方都有按 A/B/C/D 排列的 12 行。`[1,3]`以同样顺序收集源 TP 的另一条 lane。由于边界输入在源 TP 内已复制，最终 LLM TP 组 0～3 的四个 rank 都得到相同的视觉输入。

这里依赖 communicator 的明确前提：**源 TP 组内 bridge 输入必须 TP-replicated**。bridge 不负责拼接 hidden 维分片，也不做 TP all-gather；若直接传入未恢复的 TP 分片，不能期待它重建正确 embedding。[communicator 输入契约][comm-contract]。

### 展平后的第 0 维为何还能表示 batch 转换

communicator 默认把输入当作 `[B,S,H]`，但 MimoModel 显式传入 `dim_mapping={"b":0,"h":1}`，因为模态输出已经是 `[N_tokens,H]`。

因此，上例要求每条样本的视觉 token 连续排列，而且每条都是 3 行：

```text
rank0：[A0 A1 A2 B0 B1 B2]
rank2：[C0 C1 C2 D0 D1 D2]
gather：[A0 A1 A2 B0 B1 B2 C0 C1 C2 D0 D1 D2]
```

**行数可整除不等于样本边界正确。** 当前实现不传递每条样本的变长边界；尤其 FAN_OUT 对展平行数均分时，若样本视觉 token 数不一致，可能切到样本内部。不能将这一接口直接视为动态分辨率、多图片变长 batch 的通用重排器。

### 合并后才进入 LLM

[`_forward_all_modules()`][forward-all]执行 encoder 后立即调用 `_apply_colocated_comms()`，再取得文本 embedding，交给 [`align_embeddings_by_token_positions()`][align]。

本例每个 LLM DP 副本的 `input_ids` 是 `[4,16]`，其中应有 12 个视觉占位 token，其余 52 个位置是文本。视觉 `[12,H_lm]` 与文本 `[52,H_lm]`按位置合并为 `[4,16,H_lm]`，再转置成 LLM 接口的 `[16,4,H_lm]`。

之后 `_shard_language_inputs()`处理语言模型侧的序列分区。本文 CP=1；未启用 SP 时保持上述形状，若启用兼容的语言 SP，则按语言 TP 切分序列。它发生在 bridge 之后，不能据此认为共卡 bridge 支持 CP>1。

## 7. FAN_OUT 与反向传播

现在交换两侧布局：encoder TP=4/DP=2，LLM TP=2/DP=4，仍是 8 条样本、每条 3 个视觉 token。

```text
encoder ranks 0～3：A B C D → [12,H_lm]
  LLM ranks 0、1：narrow 前 6 行 → A B
  LLM ranks 2、3：narrow 后 6 行 → C D

encoder ranks 4～7：E F G H → [12,H_lm]
  LLM ranks 4、5：narrow 前 6 行 → E F
  LLM ranks 6、7：narrow 后 6 行 → G H
```

[`get_slice_info()`][comm-slice]用目标 DP 坐标计算：

```text
slot = dest_dp_idx % scale
slice_size = batch_dim_size / scale
start = slot × slice_size
```

FAN_OUT 前向是本地 `tensor.narrow(...).contiguous()`，没有 send/recv，也没有 activation all-gather。目标 rank 已经拥有 encoder 的较大 batch，只需要取自己那部分。

### 为什么 FAN_OUT backward 必须 gather

rank0 的 LLM 只算 A/B，rank2 的 LLM 只算 C/D；但两者对应的源 encoder batch 都是 A/B/C/D。若 backward 只是把本地梯度补零，就会缺少另一半样本的贡献。

[`_ColocatedCommunicate.backward()`][comm-autograd]因此在 `[0,2]`组内 all-gather `dV_AB` 和 `dV_CD`，重建完整的 `[12,H_lm]`梯度；`[1,3]`对另一条目标 TP lane 做同样的事。encoder 再按自己的 TP 布局继续反向。

FAN_IN 则相反：LLM 已经计算 A/B/C/D，rank0/1 取 A/B 梯度，rank2/3 取 C/D 梯度。这里是 narrow，**没有在 bridge backward 中对目标 TP 副本再做 reduce-scatter 或额外求和**。这种对应关系依赖模块内部 TP 的计算契约，不是任意 all-gather 的通用梯度规则。

EQUAL 两个方向都直接返回 contiguous tensor。

[`_all_gather_along_batch_dim()`][comm-gather]将所选维移动到第 0 维，调用 `all_gather_into_tensor`，再移回。当前代码按相同本地 shape 分配输出，没有长度交换或 padding 协议，因此要求 gather 组各 rank 的张量形状兼容。

## 8. 参数梯度同步与统一归一化

bridge backward 恢复的是**中间 activation 的梯度**。encoder、projector、LLM 的参数梯度仍然需要各自的 DDP 处理。[模块构建代码][model-builder]分别把模态子模块与语言模块包进 DDP，而不是给整个 MimoModel 套一层统一 DP。

以 FAN_IN 8 卡例子为例：

- encoder TP0 的参数分片在 `[0,2,4,6]` 中同步，TP1 在 `[1,3,5,7]` 中同步。
- LLM TP0 分片在 `[0,4]` 中同步，其他分片分别在 `[1,5]`、`[2,6]`、`[3,7]` 中同步。
- bridge 的 `[0,2]` 等 gather 组不承担上述参数归约。

### 8.1 先保留 loss 总和，再按同一个 token 总数缩放

本文追踪的[正确性测试 loss 函数][loss]返回：

```text
(local_loss_sum, local_valid_token_count, log_dict)
```

同时 encoder 和 LLM 的 TransformerConfig 都设为 `calculate_per_token_loss=True`。[`DistributedDataParallel`][ddp-scaling]据此将梯度缩放因子设为 1.0，并禁止 `average_in_collective=True`，使 DP 归约保留 SUM 语义。

[测试的 `_wire_training_hooks()`][wire-hooks]调用实际的 [`configure_grad_sync()`][grad-sync]，将 finalize hook 注册到 MimoModel.config，而不只在测试里模拟一套缩放公式。

标准分支的最终梯度目标是：

```text
parameter_grad = 全部有效 token 的梯度贡献之和 / N_global
```

如果先按 encoder DP=4 平均，又在统一 hook 中除 token 总数，encoder 就会多缩小一次；只按每个 rank 的本地 token 数平均，也无法在 loss mask 不均衡时得到正确的全局 token 均值。

### 8.2 finalize hook 的执行顺序

1. 调度器传入跨本次梯度累积的有效 token 数；hook 断言它非空。
2. `_global_token_count()`选择 LLM 最后 PP stage、TP rank0 的坐标，在 LLM data group 上 SUM，再从固定 source rank 向 world 广播 `N_global`。本文 PP=CP=1，data group 对应语言 DP，避免将 TP 副本重复计数。
3. 对 LLM 调用 `finalize_model_grads(..., num_tokens=None, pg_collection=language_pg)`，随后 `scale_gradients(inv)`。
4. 对每个含可训练参数的模态子模块使用自己的 PGC 做同样操作。
5. `inv = 1 / max(N_global,1)`，零 token 情况避免除零；训练 loss 的构造仍需保证无效位置不贡献梯度。

`num_tokens=None`使单模块 finalize 不再重复做一次 token 归一化。使用 distributed optimizer 且未强制 all-reduce 时，底层梯度同步走 reduce-scatter；普通分支走 all-reduce，细节见[梯度 buffer 同步][grad-buffer]。

`correct_encoder_grad_for_partial_participation`默认按 `False`读取。开启后，如果仅部分视觉 DP rank 处理模态输入，代码还会乘 `vision_dp_size / participation`；这已不是上面未经额外补偿的标准缩放。该选项也不修复共卡 collective 参与者不一致的问题，不能用它宣称任意缺失模态 batch 安全。

`overlap_grad_reduce`开启时，hook 还接入 MimoModel.no_sync；`align_grad_reduce`决定是否注册 start_grad_sync。最终归约仍必须遵循各模块自己的通信组。

### 8.3 优化器如何更新两边参数

[`get_mimo_optimizer()`][optimizer-build]按 grid map 遍历模块，仅为当前 rank 上存在且包含 `requires_grad=True`参数的模块创建优化器，并传入模块自身 PGC。完全冻结的模块不建立有效 optimizer；communicator 本身不决定哪些参数可训练。

[`MimoOptimizer.step()`][optimizer-step]先统一检查溢出，再取得各模块的梯度范数，按模块收集平方范数后汇总为整体范数，用同一整体范数裁剪，最后分别更新各模块参数。它用 world MAX 传播每个模块的范数条目，之后求平方和的平方根；不是把所有模块梯度只取一个最大值。

因此共卡异构不仅有前向特征重排，也有可微 bridge 和模块级训练接线。但联合训练的正确性仍依赖输入对齐、可训练参数、DDP 配置和统一缩放同时成立。

## 9. 支持边界与容易误读的地方

| 条件 | 当前实现含义 |
| --- | --- |
| 同一 rank span | `size`和 `rank_offset`都相同；不是部分设备重叠 |
| PP=CP=1 | communicator 明确校验，不能把共卡 TP/DP 示例推广成 PP/CP 异构 |
| 源 TP 内特征复制 | bridge 不拼接 TP hidden 分片，调用方需满足该契约 |
| DP 可整除 | 决定整数 scale；FAN_OUT 还校验所切维长度可整除 |
| 等形状 gather | 无变长长度交换与 padding；样本/视觉 token 边界由调用方维护 |
| `[N_tokens,H]`转换 | 展平行数要对应连续样本片段，均匀 token 数是当前文档契约 |
| 各 rank 模态调用一致 | forward 只处理有输入的模态；同一 gather 组若部分 rank 跳过 communicator，会有 collective 不匹配风险，属于由控制流得出的风险判断 |
| 语言侧 projection | 共卡非空配置直接拒绝；应使用模态侧 projection |
| EP/ETP/GTP | grid/模块内部可有相关机制，但本 communicator 不实现这些维度的通用跨模块重排；本文均设为 1 |
| 生命周期 | `MimoModel.destroy()`销毁其共卡 communicator；外部 topology/grid 仍由拥有方销毁 |
| Bridge 高层配置 | 当前拒绝共卡 rank 重叠；Core 能力不等于所有 Bridge recipe 均已接入 |

性能方面，本次没有测量吞吐、显存或重叠率。共卡可让模块采用不同布局，但小规模 gather、每个 rank 上两套模块状态、batch 大小变化的实际代价需要在目标模型上测量。

## 10. 已有测试证明什么，本次验证了什么

| 已有测试文件 | 代码中的检查范围 | 本次状态 |
| --- | --- | --- |
| [`test_mimo_colocated_communicator.py`][comm-tests] | rank 映射、gather 分组、非法 grid、非整除 batch、不同 batch 维位置、FAN_IN/FAN_OUT/EQUAL 前后向、资源销毁 | 已阅读，未运行 |
| [`test_mimo_colocated_correctness.py`][correctness] | 8 卡 TP2/DP4 ↔ TP4/DP2；uniform/asymmetric mask；1/4 个 microbatch；比较 encoder 更新后权重、首层梯度、LLM 输入和 logits | 已阅读，未运行 |
| [`test_mimo_1f1b_schedule.py:get_mimo_model`][model-builder] | TransformerBlock encoder + GPTModel、各自 PGC/DDP、模态侧 projection 的构建接线 | 已阅读，未运行 |

正确性测试使用 FP32、关闭 dropout 和线性 bias 等条件，并以 **encoder TP/DP 相同、LLM 调整为等 DP** 的模型为参考。虽然测试名含 `dp1_reference`，实际构建的参考并非所有模块物理 DP=1；它验证的是等效全局平均梯度，不能照名字误写参考布局。

该测试对 encoder 权重、首层梯度、LLM 输入使用 `rtol=atol=1e-3`，对不同 LLM TP 布局的 logits 使用 `1e-2`。这是测试定义的容差，不是本次运行得到的误差或通过结果。

本次完成源码版本、调用路径、8 卡 rank/样本/形状关系及文档链接的静态核对；未运行上述 GPU 测试，不报告收敛一致、吞吐提升或完整模型训练已验证。

建议按以下顺序继续读源码：

1. `test_mimo_colocated_correctness.py`：先看样本划分和对照目标。
2. `get_mimo_model()`：看配置怎样进入模块和 DDP。
3. `RankRole.build()`、`_build_colocated_communicators()`：看共卡分支与 communicator 接入。
4. `ColocatedBridgeCommunicator`：看 rank 坐标、组、切片和 autograd。
5. `_forward_all_modules()`：看特征何时重排、何时写回文本。
6. `configure_grad_sync()`、`MimoOptimizer`：看最终参数梯度和更新。

[返回 Megatron 索引](README.md) · [返回仓库首页](../README.md)

[topology-validate]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/examples/mimo/training/topology.py#L190-L223
[bridge-config]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a393057e71aa7289f34b67138352070de333d3e2/src/megatron/bridge/models/megatron_mimo/megatron_mimo_config.py#L129-L155
[bridge-example]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a393057e71aa7289f34b67138352070de333d3e2/examples/megatron_mimo/qwen35_vl/README.md#L10-L32
[omni-provider]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a393057e71aa7289f34b67138352070de333d3e2/src/megatron/bridge/models/nemotron_omni/nemotron_omni_provider.py#L243-L249
[comm-validate]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L118-L154
[model-config]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/config/base_configs.py#L12-L71
[model-builder]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py#L524-L702
[language-projection]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/model/base.py#L411-L423
[grid]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/hyper_comm_grid.py#L46-L141
[topology-build]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/examples/mimo/training/topology.py#L86-L179
[role]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/config/role.py#L71-L105
[model-init]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/model/base.py#L51-L105
[model-comm]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/model/base.py#L1092-L1131
[comm-init]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L56-L116
[comm-mapping]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L162-L199
[comm-tests]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_communicator.py#L55-L190
[data-slice]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py#L307-L337
[forward-step]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py#L109-L153
[comm-contract]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L41-L54
[forward-all]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/model/base.py#L1133-L1225
[align]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/model/base.py#L275-L367
[comm-slice]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L209-L245
[comm-autograd]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L258-L304
[comm-gather]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/comm/colocated_communicator.py#L307-L325
[loss]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py#L79-L106
[ddp-scaling]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/distributed/distributed_data_parallel.py#L249-L255
[wire-hooks]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py#L168-L186
[grad-sync]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/examples/mimo/training/grad_sync.py#L106-L210
[grad-buffer]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/distributed/param_and_grad_buffer.py#L748-L784
[optimizer-build]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/optimizer.py#L393-L442
[optimizer-step]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/megatron/core/models/mimo/optimizer.py#L61-L129
[correctness]: https://github.com/NVIDIA/Megatron-LM/blob/07147d6942dfd4e6d1566b64bb1730f478a1bc32/tests/unit_tests/models/mimo/test_mimo_colocated_correctness.py#L857-L1120
