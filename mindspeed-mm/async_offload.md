# MindSpeed-MM 异步激活卸载机制

异步激活卸载把暂时不需要的激活从设备搬到 CPU，释放设备存储，并在反向或重计算需要这些数据时恢复。它通过拷贝流和计算流之间的事件依赖控制先后顺序，并尝试提前取回下一层将使用的张量，使数据搬运与计算重叠。

本文围绕插件式 FSDP2 的 legacy 异步激活卸载机制逐步展开，目前介绍单张量搬运组件 `SwapTensor` 和按层管理组件 `OffloadManager`。配置入口及 `saved_tensors_hooks` 的完整接入过程暂不展开。

| 项目 | 范围 |
| --- | --- |
| 源码仓库 | [Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| 分析版本 | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` |
| 记录日期 | 2026-10-09 |
| 主源文件 | `mindspeed_mm/fsdp/features/memory/async_offload.py` |
| 适用范围 | 插件式 FSDP2 的 legacy 激活卸载；设备接口适配 CUDA/NPU，不限定具体模型 |
| 并行范围 | 进程内设备与 CPU 之间的搬运，不执行跨 rank 通信；本文不验证特定 DP/TP/CP/PP 组合 |
| 验证方式 | 本地源码静态分析，未运行设备拷贝、训练或数值对齐实验 |

分析时本文引用的主源文件、设备工具、单例工具和 Flash Attention 调用文件没有未提交修改。源码链接固定到上述提交，路径及行号已按本地文件核对，未验证远端页面可访问性。PyTorch 与 torch_npu 的 storage、事件及分配器底层实现未在本文中检查。

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
