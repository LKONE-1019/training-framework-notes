# MindSpeed-MM 异步激活卸载机制

异步激活卸载把暂时不需要的激活从设备搬到 CPU，释放设备存储，并在反向或重计算需要这些数据时恢复。它通过拷贝流和计算流之间的事件依赖控制先后顺序，并尝试提前取回下一层将使用的张量，使数据搬运与计算重叠。

本文分析插件式 FSDP2 的 legacy 异步激活卸载机制，介绍单张量搬运组件 `SwapTensor`、按层管理组件 `OffloadManager`，并通过 `async_save_on_cpu` 串联配置接入、前向保存、跨层释放、反向恢复与预取。

| 项目 | 范围 |
| --- | --- |
| 源码仓库 | [Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| 分析版本 | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` |
| 记录日期 | 2026-10-09 |
| 主源文件 | `mindspeed_mm/fsdp/features/memory/async_offload.py` |
| 适用范围 | 插件式 FSDP2 的 legacy 激活卸载；设备接口适配 CUDA/NPU，不限定具体模型 |
| 并行范围 | 进程内设备与 CPU 之间的搬运，不执行跨 rank 通信；本文不验证特定 DP/TP/CP/PP 组合 |
| 验证方式 | 本地源码静态分析，未运行设备拷贝、训练或数值对齐实验 |

分析时本文引用的主源文件、特性接入文件、参数定义、设备工具、单例工具和 Flash Attention 调用文件没有未提交修改。源码链接固定到上述提交，路径及行号已按本地文件核对，未验证远端页面可访问性。PyTorch 与 torch_npu 的 storage、事件及分配器底层实现未在本文中检查。

## 1. 组件分工

异步卸载需要解决三个问题：哪些张量需要保存、何时释放或恢复、如何实际搬运。当前实现将这些职责分开：

| 组件 | 职责 |
| --- | --- |
| `async_save_on_cpu` / 算子调用方 | 在保存、恢复激活的时机调用卸载组件，决定张量筛选与使用顺序 |
| `OffloadManager` | 按层编号登记张量，触发上一层存储释放、反向预取，以及显式清理登记项 |
| `GetCnt` / `OffloadItem` | 分别提供层内编号与计数，以及单个登记项的数据结构 |
| `SwapTensor` | 持有单张量的设备引用、CPU 副本和事件，执行 D2H、storage 释放及 H2D |

```text
前向保存激活 → 生成 key → SwapTensor 提交 D2H → OffloadManager 登记
后续层触发释放 → 按上一层 key 前缀查找 → SwapTensor 释放设备 storage
反向需要激活 → SwapTensor 提交 H2D → 计算流等待事件 → 清理登记并预取前一层
```

D2H 指 device-to-host，H2D 指 host-to-device。该机制针对激活数据，不管理参数分片或优化器状态。计算与搬运能否有效重叠取决于实际调度和负载，本文没有性能实测。

## 2. SwapTensor：单张量的数据搬运

[`SwapTensor`][swap-tensor] 保留原 Tensor 对象，通过缩小、恢复底层 storage 控制设备内存占用。Tensor 的 shape、stride 等元信息与底层 storage 是不同概念，释放 storage 不等于销毁 Tensor 对象。

### 2.1 成员变量

| 成员变量 | 含义与用途 |
| --- | --- |
| `tensor` | 原设备张量的引用，不创建设备副本；H2D 时写回这个张量 |
| `size` | 保存原 shape，当前类中没有进一步使用 |
| `storage_size` | 原 typed storage 的元素数量，单位不是字节；用于恢复容量 |
| `tensor_cpu` | 与原张量 shape、dtype 相同的 CPU pinned-memory 缓冲区，初始化时尚无有效副本 |
| `is_slice_tensor` | 判断 `storage().size() != numel()`，据此选择拷贝方式；不是通用的 view 或非连续张量判定 |
| `stat` | 调度状态，初始为 `"device"`；提交 D2H 后为 `"host"`，提交 H2D 后为 `"device"` |
| `key` | 调用方提供的标识，例如 `"2_0"`；类内部不解析它 |
| `d2h_event` / `h2d_event` | 分别在 D2H / H2D 拷贝之后记录，供使用方建立流间等待依赖 |

### 2.2 成员函数与生命周期

| 函数 | 执行过程 |
| --- | --- |
| `__init__(tensor, key)` | 保存元信息、分配 CPU 页锁定内存并创建两个事件；不执行拷贝 |
| [`launch_d2h(stream)`][launch-d2h] | 仅在 `stat == "device"` 时执行。在当前流记录 `forward_event`，拷贝流等待它后复制数据到 CPU，记录 `d2h_event`，设为 `"host"`；此时尚未释放设备 storage |
| [`wait_d2h_finished()`][wait-d2h] | 仅在 `stat == "host"` 时执行。让当前流等待 `d2h_event`，调用 `tensor.storage().resize_(0)`，状态仍为 `"host"` |
| [`launch_h2d(h2d_stream)`][launch-h2d] | 仅在 `stat == "host"` 时执行。让 H2D 流等待当前流的 `backward_event`，恢复 storage 容量并拷回数据，记录 `h2d_event`，设为 `"device"` |

两次拷贝都在 `torch.no_grad()` 下使用 `non_blocking=True`。当 `is_slice_tensor=True` 时，通过 Tensor 的 `copy_()` 复制逻辑元素；否则通过 storage 的 `copy_()` 复制完整物理存储。CPU pinned memory 为异步传输提供条件。

```text
device → 提交 D2H → host（设备 storage 尚在）
       → 建立 D2H 等待依赖并缩小 storage → host（设备 storage 容量为 0）
       → 恢复 storage 并提交 H2D → device → 使用方计算流等待 h2d_event 后使用
```

`stat` 表示已提交到哪个阶段，不能证明拷贝已完成；`wait_event()` 建立设备流依赖，也不等于阻塞 Python 线程的 `synchronize()`。当前常规调用让 D2H、H2D 共用一条 `swap_stream`，两次拷贝按提交顺序执行；`launch_h2d()` 自身没有显式等待 `d2h_event`，换成不同拷贝流时需要另行保证依赖。

释放 storage 会影响共享它的其他 view，切片分支也只保存该切片可见的数据。调用方必须保证释放时其他使用者已不再需要原设备数据，具体跨流释放安全性依赖运行时和分配器。恢复后的底层内存地址不保证与原地址相同；CPU 副本继续由对象持有。设备缓存分配器还可能保留已释放的内存，因此 storage 缩小不保证 reserved memory 立即下降。

## 3. OffloadManager：按层登记、释放与预取

[`OffloadManager`][manager] 把多个 `SwapTensor` 组织起来，使调用方能够按层找到张量、释放上一层的设备存储，并在反向时预取下一步需要的数据。它管理登记关系和调度，实际搬运仍由 `SwapTensor` 执行。

### 3.1 单例与成员变量

`OffloadManager(metaclass=Singleton)` 使用[按类缓存实例的单例元类][singleton]。同一进程反复调用 `OffloadManager()` 会取得同一个对象，首次创建时才执行 `__init__()`。不同训练进程各有自己的实例；它不是跨 rank 共享的全局对象，也没有按模型、microbatch 或设备建立独立命名空间。

| 成员变量 | 含义与用途 |
| --- | --- |
| `items` | 字典，保存 `key → OffloadItem`，其中 `act` 在主要调用路径中是 `SwapTensor` |
| `check` | 保存构造参数 `check=False`；当前实现没有用它控制校验 |
| `npu_item` | 用列表实现的后进先出栈，保存暂留设备的对象，主要用于最后一层的算子缓存 |
| `getcnt` | `GetCnt` 实例，生成层内编号并记录各层的计数 |
| `swap_stream` | 初始化时创建的拷贝流，当前常规调用中 D2H 和 H2D 共用它 |

[`OffloadItem`][item] 只保存三个字段：`act`、`ref_cnt` 和 `event`。其中 `event` 是 `put()` 可选传入的管理项事件，不能与 `SwapTensor.d2h_event` / `h2d_event` 混为一谈；主要调用路径使用默认值 `None`。

```text
items["2_0"] → OffloadItem(act=SwapTensor(...), ref_cnt=1, event=None)
items["2_1"] → OffloadItem(act=SwapTensor(...), ref_cnt=1, event=None)
npu_item     → [SwapTensor(...), SwapTensor(...), ...]
```

### 3.2 编号、登记与查询

`get_cnt(block_idx)` 委托 [`GetCnt.get_cnt()`][counter] 返回 `(key, after_block)`。key 的格式是 `block_idx_tensor_idx`，例如 `"2_1"` 表示 offload block 2 的第 1 个编号；层号来自调用方，不应直接当作模型原始层号。

| 情况 | 计数器行为 |
| --- | --- |
| 层号增大 | 新层计数设为 1，返回层内编号 0；若层号不为 0，则 `after_block=True` |
| 层号不变 | 当前层计数加 1，返回下一个层内编号 |
| 层号减小 | 按新一轮处理，丢弃旧计数表，只保留当前层计数 1；不会清理管理器里的对象 |

这里统计的是 `get_cnt()` 调用次数，不是字典当前长度，也不是按 Tensor 身份去重后的数量。例如最后一层可能生成编号后直接保留原张量，没有向 `items` 登记。

[`put(key, act, event=None)`][put] 在新 key 下创建 `ref_cnt=1` 的登记项；若 key 已存在，则覆盖 `act` 和 `event`，同时增加计数。它不主动启动 D2H。常规前向调用先执行 `SwapTensor.launch_d2h()`，再登记对象。

`exist(key)` 返回字典是否包含 key；`assert_exist(key)` 在缺失时抛出 `RuntimeError`。`get_layer_items_keys(block_idx)` 根据该层记录的数量生成候选 key，再过滤出仍在字典中的项，返回结果按层内编号升序排列。这使算子能一次取回一组缓存，但也要求计数信息与登记过程保持一致。

### 3.3 跨层释放设备存储

[`del_npu_tensor(prefile_key)`][release] 遍历 `items`，对前缀匹配的项调用 `act.wait_d2h_finished()`。例如传入 `"2_"`，就处理第 2 层的登记张量。

尽管方法名是 `del_npu_tensor`，它不删除 `items` 项，也不操作 `npu_item` 栈；它把设备 storage 的释放交给 `SwapTensor`。对象及 CPU 副本仍需保留，供后续恢复。

在[前向 pack 调用处][pack]，进入更大层号且第一次生成符合条件的张量编号时，`after_block=True` 触发 `del_npu_tensor(f"{block_idx - 1}_")`。因此，D2H 提交和 storage 释放分属两个时机；释放也不是独立的层结束 hook，而是由后续层的登记动作触发。

### 3.4 获取对象、显式清理与设备缓存栈

[`get(key)`][get] 检查 key 存在后取出 `act`，将 `ref_cnt` 减 1 并返回。它不启动 H2D，也没有实现注释所说的“引用计数归零自动删除”。因此实际使用需要区分以下动作：

```text
get(key) → 取得 SwapTensor
launch_h2d(stream) → 提交恢复
计算流等待 h2d_event → 后续计算可使用恢复数据
clear(key) → 删除管理器的登记关系
```

`clear(key)` 无条件删除指定登记项，不检查计数；key 不存在时抛错。`clear()` 无参数时仅清空 `items`，不会清空 `npu_item`、重置 `GetCnt` 或同步流。删除登记关系也不等于立即销毁 Tensor，其他引用仍可持有对象。

上述流程适用于显式按 key 取回的算子调用。[autograd unpack 路径][unpack] 则已经拿到 pack 返回的 `SwapTensor`，可以直接恢复后 `clear(key)`，无需再调用 `get()`。

`put_npu_tensor(act)` 与 `pop_npu_tensor()` 分别执行列表 `append()`、`pop()`，不按 key 查询，也不做传输或同步。在 [SkipRecomputeFlashAttention][flash-cache] 中，最后一层依次保存 `attn_output`、`softmax_max`、`softmax_sum`，重计算阶段按相反顺序弹出，以复用结果。这与普通 pack 路径最后一层直接返回原 Tensor 是不同的保存方式。

### 3.5 反向预取

[`prefetch_get(block_idx, tensor_idx, h2d_stream, d2h_stream)`][prefetch] 向 `GetCnt` 请求候选 key，对仍存在的项依次调用 `get()`，执行 `d2h_stream.wait_stream(h2d_stream)`，然后调用 `launch_h2d(h2d_stream)`。预取保留登记项，实际消费者使用时再清理。

前向层序为 `0 → 1 → 2` 时，反向通常沿 `2 → 1 → 0` 进行。因此在恢复第 1 层的数据后，提前提交第 0 层 H2D，就有机会让它与第 1 层计算重叠。当前主要调用中两个流参数都是同一条 `swap_stream`；不能把这里的 `wait_stream()` 解读成已提供两条独立传输流之间的完整依赖方案。

以下例子只说明单进程调度：总卡数 1，DP/TP/CP/PP 均为 1，一个 microbatch 顺序经过 3 个 block，每层有 2 次符合条件的编号动作。是否发生这些保存动作以实际计算图为准，例子不依赖具体张量 shape。

| 时机 | `items` 与搬运行为 |
| --- | --- |
| 前向 block 0 | `"0_0"`、`"0_1"` 提交 D2H 并登记 |
| 前向 block 1 首次符合条件的保存 | 触发 `"0_"` 的 storage 释放；本层 `"1_0"`、`"1_1"` 提交 D2H 并登记 |
| 前向 block 2 首次符合条件的保存 | 触发 `"1_"` 的 storage 释放；最后一层普通 pack 路径直接保留 Tensor |
| 普通 unpack 处理 block 2 | 输入是 Tensor，直接返回，不在该分支启动预取 |
| 普通 unpack 首次处理 block 1 的卸载对象 | 按需恢复当前对象、清理其登记，并提交 block 0 的 H2D 预取 |
| 普通 unpack 处理 block 0 | 已预取对象的 `launch_h2d()` 因状态为 `"device"` 而直接返回，计算流仍需等待既有 `h2d_event`；使用时清理登记 |

因此，普通 hook 路径并没有在最后一层返回 Tensor 时自动预取倒数第二层；部分显式算子缓存路径另有预取调用，不能混为一条调度序列。

### 3.6 当前实现的假设与不一致

以下结论来自当前源码，区分设计意图与已实现行为，不代表已复现设备运行问题：

- **引用计数不控制释放。** `get()` 只递减，`clear()` 才删除；重复预取仍调用 `get()`，即使 `launch_h2d()` 不再提交拷贝，也可能使计数降到负数。不能用 `ref_cnt` 判断真实剩余消费者数量。
- **预取 key 生成存在限制。** [`GetCnt.get_prefetch_keys()`][prefetch-keys] 使用当前层的数量生成 `block_idx - 1` 的 key。当前层有 2 项、上一层有 3 项时，上一层编号 2 的项不会被预取，需实际消费时恢复。函数算出了较小层号中的最大值，却只用来判断是否存在前层；层号有空洞时仍会找 `block_idx - 1`。`tensor_idx` 也未参与计算。
- **计数器依赖顺序。** 层号回退被解释为新一轮，key 又不含 iteration 或 microbatch 标识；若旧对象未消费完就开始交错登记，可能覆盖旧 key。不能据此认定支持任意交错前向或重入执行。
- **断言条件写反。** `assert_not_exist()` 在 key 不存在时抛出“already exist”，与命名不符；当前文件没有调用它。
- **可选事件分支不完整。** `get()` 在 `event is not None` 时调用 `item.get_event().wait()`，但 `OffloadItem` 未定义 `get_event()`；当前普通调用传入 `None`，不会进入该分支。

这些限制应在扩展调度、接入不同算子或修改流配置时单独核实。本文没有执行训练、内存压力或数值一致性验证。

## 4. 激活卸载的完整流程

普通模块接入路径由 `saved_tensors_hooks` 串起前向保存与反向读取：前向保存符合条件的激活时提交 D2H，后续层触发设备 storage 释放，反向读取时提交 H2D，并尝试提前取回更早一层的激活。本章解释 `async_offload.py` 的 legacy 路径；`impl="stash"` 会走 `ActStash`，其普通激活保存由共享 swap cache 管理，不套用本章调度。

```text
配置选中模块 → 包装 forward → 进入 saved_tensors_hooks 上下文
                                   ↓
                 autograd 保存反向所需张量 → pack 筛选
                                   ↓
               非最后一层：SwapTensor 提交 D2H 并登记
                                   ↓
            后续层首次符合条件的保存 → 释放上一层 storage
                                   ↓
            反向读取保存对象 → unpack 恢复 storage 并提交 H2D
                                   ↓
        计算流等待 H2D 事件 → 清理登记、提交前一层预取 → 返回 Tensor
```

### 4.1 配置与模块接入

参数见 [`FeatureArguments.enable_activation_offload`][offload-enable] 和 [`ActivationOffloadPlanConfig`][offload-plan]：

| 配置项 | 默认值 | 生效作用 |
| --- | --- | --- |
| `enable_activation_offload` | `False` | 是否应用激活卸载包装 |
| `activation_offload_plan.apply_modules` | `None` | 选择要包装的模块；未配置时入口直接返回 |
| `activation_offload_plan.impl` | `"legacy"` | 选择 legacy 或 stash 实现 |

[`FeaturesApplier.apply_activation_offload_modules()`][offload-entry] 检查开关和模块配置后，在 legacy 分支调用 `get_offload_modules()`，再调用 `async_offload_modules()`。

[`get_offload_modules()`][offload-modules] 根据模块路径匹配目标。带 `{*}` 的路径会定位对应的可迭代层容器并遍历；其他路径通过 `named_modules()` 与 `module_name_match()` 匹配。返回项包含模块名称、模块对象、卸载层编号和总层数。

`layer_idx` 按匹配列表顺序编号，`depth` 最终统一为匹配模块总数；它们不是必然与模型原始层号、总深度相同。后续跨层释放和预取依赖这些编号与实际执行顺序相符。

[`async_offload_modules()`][offload-wrap] 替换目标模块的 `forward`：

```python
module.forward = with_async_save_on_cpu(name, layer_idx, depth)(module.forward)
```

### 4.2 forward 包装与保存钩子

[`with_async_save_on_cpu()`][offload-context] 默认取第一个位置参数作为 `hidden_states`，构造 `async_save_on_cpu` 上下文，更新 `TrainingContext` 的模型深度和当前层号，然后在该上下文中执行原 `forward`。

```python
context = async_save_on_cpu(
    block_idx=layer_idx,
    depth=depth,
    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
    prefetch=prefetch,
)
with context:
    return forward_func(*args, **kwargs)
```

[`async_save_on_cpu`][save-hooks] 继承 PyTorch 的 `saved_tensors_hooks`，在构造时注册 `_pack_to_cpu` 和 `_unpack_from_cpu`：

| 参数或钩子 | 作用 |
| --- | --- |
| `block_idx` | 当前模块的卸载层编号，用于生成 key 和选择前一层 |
| `depth` | 匹配模块总数，用于判断最后一层 |
| `custom_check_fn` | 在基础筛选后追加张量筛选；类本身默认不添加这个条件，模块包装会添加输入指针条件 |
| `prefetch` | 默认 `True`，控制 unpack 是否调用前一层预取 |
| `_pack_to_cpu` | autograd 保存反向所需张量时调用，返回原 Tensor 或 `SwapTensor` |
| `_unpack_from_cpu` | autograd 读取保存对象时调用，返回可用于后续计算的 Tensor |

进入 `forward` 只是安装保存钩子，实际搬运由 autograd 的保存动作触发；它不在模块返回时自动复制所有输出。退出上下文后，已经保存的对象仍可在反向读取时通过对应 unpack 恢复。

默认包装要求输入在 `args[hidden_states_idx]` 中，`hidden_states_idx=0`。若位置参数缺失或没有 `data_ptr()`，就发出警告并直接执行原 `forward`；只通过 `hidden_states=...` 关键字传入时，不会自动从 `kwargs` 获取输入。

### 4.3 前向 pack：筛选、复制与登记

[`_pack_to_cpu()`][pack] 首先执行 [`base_check_fn()`][base-check]，排除 `Parameter`、直接以 `Parameter` 为 `_base` 的 view，以及空 storage 张量；再执行自定义条件。默认模块包装要求保存张量的 `data_ptr()` 与输入 `hidden_states` 相同。

因此，默认接入主要针对被 autograd 保存的模块输入激活，不覆盖所有中间结果。同一输入被多个算子分别保存时，也可能产生多次 pack 和多个 key；计数器没有按张量身份去重。

符合条件后，pack 生成 `block_idx_tensor_idx` 形式的 key，并在需要时先释放上一层 storage（见第 4.4 节）。若当前是最后一个匹配模块，直接返回原 Tensor；否则执行：

```python
swap_tensor = SwapTensor(tensor, key)
swap_tensor.launch_d2h(OffloadManager().swap_stream)
OffloadManager().put(key, swap_tensor)
return swap_tensor
```

`SwapTensor` 分配 CPU pinned-memory 缓冲区。`launch_d2h()` 在当前计算流记录 `forward_event`，让拷贝流等待此前计算，然后非阻塞复制到 CPU 并记录 `d2h_event`。autograd 保存 pack 返回的包装对象，管理器也登记它，供跨层释放和预取查询。

提交 D2H 后，原设备 storage 仍然存在，CPU 缓冲区与设备存储会暂时同时占用内存。`stat="host"` 只表示搬运已提交，不能据此读取尚未完成的 CPU 副本。

### 4.4 跨层释放：后续保存动作触发

在更大层号第一次生成符合条件的 key 时，`GetCnt.get_cnt()` 返回 `after_block=True`，pack 调用：

```python
OffloadManager().del_npu_tensor(f"{block_idx - 1}_")
```

管理器按 key 前缀遍历上一层登记项，调用 `wait_d2h_finished()`。该方法让当前流等待 `d2h_event`，随后执行 `tensor.storage().resize_(0)`；Tensor 对象、shape 等元信息和 CPU 副本继续保留。

这将数据复制与设备存储释放分成两个时机，为搬运与前向计算重叠提供机会。释放不是独立的层结束 hook：如果后续层没有符合条件的保存，就不会出现这次触发；层号有空洞时，代码也仍只查找 `block_idx - 1`。

最后一层的直接返回发生在跨层释放之后，因此最后一层虽不卸载自身输入，仍能在符合条件的保存时触发倒数第二层 storage 释放。保留最后一层输入是为了避免紧接着反向时又立刻搬回。

`wait_event()` 建立设备流依赖，不是 Python 线程上的 `synchronize()`。storage 释放涉及共享 view 以及分配器对异步使用的处理，运行时前提见第 2.2 节；本文未验证底层跨流释放行为。

### 4.5 反向 unpack：恢复当前对象并预取前一层

[`_unpack_from_cpu()`][unpack] 收到原 Tensor 时直接返回，包括最后一层以及未通过筛选的张量。这个分支不启动预取。

收到 `SwapTensor` 时，unpack 已经持有保存对象，不必先通过管理器 `get()` 查询。执行顺序是：

1. 调用 `launch_h2d()`，让拷贝流等待当前计算流的事件，恢复原 storage 容量，将 CPU 副本写回并记录 `h2d_event`。
2. 当前计算流等待 `h2d_event`，保证后续反向计算能够读取恢复的数据。
3. 从 key 解析层号和层内编号，调用 `clear(key)` 删除登记项，更新 `TrainingContext` 当前层号。
4. 若 `prefetch=True`，调用 `prefetch_get()` 提交前一层的 H2D，最后返回原 `swap_tensor.tensor`。

前向层序为 `0 → 1 → 2` 时，反向通常沿 `2 → 1 → 0` 进行。恢复层 1 当前激活后提前搬回层 0，就有机会让层 0 的 H2D 与层 1 的反向计算重叠。

预取对象继续留在 `items` 中，直到实际 unpack 清理。已经提交 H2D 的对象状态为 `"device"`，再次调用 `launch_h2d()` 会直接返回，但实际消费时计算流仍需等待此前记录的 `h2d_event`。预取并不保证数据已经到达。

普通 hook 路径的 D2H、H2D 共用 `OffloadManager.swap_stream`；它提供一条与计算流独立的搬运流，并非两条独立的 D2H/H2D 流。当前预取按当前层计数构造前一层候选 key，可能遗漏前一层多出的对象，细节见第 3.6 节。

### 4.6 三层模型的数据流示例

以下是调度示例，不是训练实测：总卡数 1，rank 0，DP/TP/CP/PP 均为 1；一次前向与反向处理一个 microbatch，batch size 为 1。三个被包装的 block 按 `0 → 1 → 2` 执行，输入分别为 `x0`、`x1`、`x2`，均假设为独立、连续的 BF16 张量，shape 为 `[1, 4, 4]`。每个输入包含 16 个元素，逻辑数据量为 32 字节，不代表实际分配器占用。

为突出时序，假设每层恰好发生一次通过筛选的输入保存，没有重计算或其他显式算子缓存，并且后续层释放时不再需要上一层输入的设备数据。

| 时机 | autograd 保存对象 | 搬运、设备存储与管理器变化 |
| --- | --- | --- |
| 前向 block 0 保存 `x0` | `SwapTensor(x0, "0_0")` | 提交 `x0` D2H，登记 `"0_0"`；设备 storage 尚保留 |
| 前向 block 1 保存 `x1` | `SwapTensor(x1, "1_0")` | 触发 `x0` storage 缩小到 0；提交 `x1` D2H，登记 `"1_0"` |
| 前向 block 2 保存 `x2` | 原 Tensor `x2` | 触发 `x1` storage 缩小到 0；`x2` 留在设备，不登记 `"2_0"` |
| 反向 block 2 读取 `x2` | 直接返回 `x2` | 使用现有设备数据；该分支不预取 `x1` |
| 反向 block 1 读取 `x1` | 返回恢复后的 `x1` | 按需恢复 `x1`，计算流等待其 H2D 事件，清理 `"1_0"`；提交 `x0` H2D 预取 |
| 反向 block 0 读取 `x0` | 返回恢复后的 `x0` | 复用已经提交的 H2D，计算流等待其事件，清理 `"0_0"` |

在该示例中，层 1 的首次回搬是按需发生的，层 0 才有机会通过预取隐藏搬运耗时。如果预取尚未完成，层 0 的反向计算仍需等待；实际收益取决于搬运带宽、计算时长和执行调度。

### 4.7 阅读与验证边界

该路径改变反向所需激活的保存方式，并在使用前恢复数据。D2H/H2D 位于 `torch.no_grad()` 中，不为拷贝建立额外求导图；本章没有验证恢复数据的数值一致性、共享 storage 情形或特定模型训练结果。

激活卸载本身不执行跨 rank 梯度同步，也不管理参数分片、参数梯度或优化器状态。显式算子缓存路径可能在重计算阶段调用同一组组件，但其保存和预取触发点不同；第 3.4 节的 Flash Attention 缓存不能直接套用第 4.6 节的普通 hook 时序。

扩展到交错 microbatch、重入前向或不同拷贝流时，应结合第 3.6 节检查编号、登记和预取假设。本文仅覆盖当前源码的模块包装及普通 saved-tensor 调用流程，没有运行训练、性能或设备内存验证。

[swap-tensor]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L67
[launch-d2h]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L85
[wait-d2h]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L102
[launch-h2d]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L110
[manager]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L139
[singleton]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/utils/decorators.py#L1
[item]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L128
[counter]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L36
[put]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L166
[release]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L184
[get]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L200
[pack]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L242
[unpack]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L268
[flash-cache]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/flash_attn/skip_recompute_flash_attn.py#L55
[prefetch]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L212
[prefetch-keys]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L53
[offload-entry]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/apply_features.py#L73
[offload-plan]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/params/feature_args.py#L190
[offload-enable]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/params/feature_args.py#L256
[offload-modules]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L291
[offload-wrap]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L325
[offload-context]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L331
[save-hooks]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L228
[base-check]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/features/memory/async_offload.py#L17
