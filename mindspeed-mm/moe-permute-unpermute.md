# MindSpeed-MM MoE Token 重排：eager 流程与 NPU 融合接口

MoE 路由通常按 token 保存 Top-K 专家编号，但专家计算需要同一专家的输入连续排列。`permute` 完成复制与分组；专家计算结束后，`unpermute` 恢复各 token 的路由槽位，并按路由权重合并输出。它们自身不执行专家 GEMM，也不发起跨设备通信。

| 项目 | 分析范围 |
| --- | --- |
| 上游仓库 | [Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| 完整源码 commit | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` |
| 分析日期 | 2026-10-15 |
| 代码范围 | 插件式 FSDP2 的 `ops/moe_ops/permute.py`、`unpermute.py`，以及 Kimi K3、EP dispatcher 的调用处 |
| 适用范围 | 使用上述公共算子的 MoE 路径；融合分支需要 NPU 与配套 `torch_npu`，具体设备和版本能力由外部依赖决定 |
| 示例并行配置 | 单设备，EP/TP/CP/DP/PP 均为 1；一个 microbatch，B=1、S=3，因此 T=3；K=2。EP 调用另行说明 |
| 验证状态 | 源码静态分析，并运行独立 eager 函数的 CPU 小规模前向与梯度检查；未运行 NPU 融合、完整模型或分布式训练验证 |
| 工作区差异 | 分析涉及的源文件与上述 commit 一致；源码工作区其他未跟踪文件未纳入分析 |

源码链接固定到上述 commit，文件、函数和行号已在本地核对，未逐页验证 GitCode 页面可访问性。融合接口另参考文末标明版本的外部资料，不能据此认定本机安装了对应运行时。

## 1. 核心结论与接口选择

- 一对 `eager_permute → eager_unpermute` 共调用三次 `argsort`：第一次按专家排序，后两次在正向排列 `p` 和逆置换 `q` 之间转换。
- `permute` 的第二个返回值是 `q`：原路由槽位在专家分组结果中的位置。它既不是专家编号，也不是原 token 编号。
- `unpermute` 先恢复 `[T*K,D]` 的路由槽位，再变为 `[T,K,D]`，乘权重并沿 K 求和，得到 `[T,D]`。
- 当前 eager 的 `restore_shape` 指定合并前缓冲区，并截取输入前若干行；不能直接当作最终输出形状，也不是任意裁剪后的通用补零机制。

公共入口的默认值是 `fused=True`，但实际选择需要同时满足 `IS_NPU_AVAILABLE`。见 [`permute()`][permute-entry] 与 [`unpermute()`][unpermute-entry]：

```python
if fused and IS_NPU_AVAILABLE:
    return fused_permute(tokens, indices, num_out_tokens)
else:
    return eager_permute(tokens, indices, num_out_tokens)
```

因此，`fused=True` 表示请求融合分支，不意味着 CPU 上会调用 NPU 算子。要明确阅读 eager 行为，应以公共入口的 `fused=False` 为准。

## 2. 张量、路由槽位与两个索引

令 T 为输入 token 数，D 为隐藏维度，K 为每个 token 选择的专家数，N=T*K。普通未截断路径使用以下张量：

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| `tokens` | `[T,D]` | 每行一个输入 token 的向量 |
| `indices` | `[T,K]`，或 K=1 时 `[T]` | 每个 token 的目标专家编号 |
| `probs` | `[T,K]` | 原 token、原 Top-K 顺序下的路由权重 |
| `p` | `[N]` | 专家分组后第 j 行来自原路由槽位 `p[j]` |
| `q` | `[N]` | 原路由槽位 i 位于专家分组结果的第 `q[i]` 行 |
| `permuted_tokens` | `[N,D]` | 按专家分组的输入；专家计算后，同样顺序下保存专家输出 |

“路由槽位”表示一次 token 到专家的分配。一个 token 选 K 个专家，就有 K 个槽位；槽位 i 所属的 token 编号是 `i // K`。

贯穿全文的例子是：

```python
tokens = [A, B, C]  # 每个字母代表一个 D 维向量
indices = [[2, 0], [1, 2], [0, 1]]
probs = [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]
```

| 展平槽位 i | token | 专家编号 | 权重 |
| --- | --- | --- | --- |
| 0 | A | 2 | 0.7 |
| 1 | A | 0 | 0.3 |
| 2 | B | 1 | 0.4 |
| 3 | B | 2 | 0.6 |
| 4 | C | 0 | 0.8 |
| 5 | C | 1 | 0.2 |

## 3. eager_permute：从 token 顺序变为专家分组顺序

实现见 [`eager_permute()`][eager-permute]。首先确定 K，保存专家索引的 dtype，再对展平的专家编号稳定排序：

```python
topk = 1 if indices.dim() == 1 else indices.size(1)
indices_dtype = indices.dtype
sorted_indices = torch.argsort(indices.float().view(-1), stable=True)
```

`argsort` 返回的是排序所需的位置，而不是排序后的专家编号。本文把这里的 `sorted_indices` 记作 p：

```text
专家编号      [2, 0, 1, 2, 0, 1]
p             [1, 4, 2, 5, 0, 3]
专家编号[p]   [0, 0, 1, 1, 2, 2]
```

`stable=True` 保持相同专家内部的原槽位顺序。例如专家 0 对应槽位 1 和 4，排序后仍然是 1 在 4 前面。

### 3.1 为什么要除以 topk

下一步是：

```python
permuted_tokens = tokens.index_select(0, sorted_indices // topk)
```

p 是展平后的路由槽位编号，而 `tokens` 只有 T 行。由于每个 token 连续占 K 个槽位，`p // K` 把槽位转换为原 token 行号：

```text
p       = [1, 4, 2, 5, 0, 3]
p // 2  = [0, 2, 1, 2, 0, 1]
输出    = [A, C, B, C, A, B]
          专家0  专家1  专家2
```

`index_select(0, ...)` 按指定顺序读取整行向量。重复的索引使同一个 token 出现多次，因此形状由 `[T,D]` 变成 `[T*K,D]`。输入 token 的重复发生在这一步，不是由后面的专家 GEMM 隐式完成。

### 3.2 返回逆置换 q

```python
sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True).to(indices_dtype)
```

这里对 p 求逆，得到 q：

```text
p = [1, 4, 2, 5, 0, 3]
q = [4, 0, 2, 5, 1, 3]
```

p 是 0 到 N-1 的排列，所以 `q[p[j]]=j`、`p[q[i]]=i`。第二次 `argsort` 只是算出逆置换，没有再次重排 token；`permuted_tokens` 此时仍然是 `[A,C,B,C,A,B]`。

最后按需截断并返回：

```python
if num_out_tokens:
    permuted_tokens = permuted_tokens[:num_out_tokens]
return permuted_tokens, sorted_indices1
```

当前实现先生成完整的 `[N,D]`，再切片；它不会因设置较小的 `num_out_tokens` 而省去前面完整的 `index_select`。第二个返回值 q 仍然保留 N 个元素。

## 4. eager_unpermute：恢复槽位并加权合并

实现见 [`eager_unpermute()`][eager-unpermute]。它接收的 `sorted_indices` 是前一节返回的 q，名称与 `eager_permute` 内部第一次排序的变量相同，含义却不同。

令 `A2=Expert2(A)`，其余符号类似。专家计算保持分组行顺序，输出为：

```text
Y = [A0, C0, B1, C1, A2, B2]
```

### 4.1 确定缓冲区大小

有 `probs` 时，源码取：

```python
num_tokens = probs.numel()  # 实际是 T*K 个槽位
topk = probs.size(1)
```

`restore_shape=None` 时分配 `[T*K,D]` 的零缓冲区。没有 `probs` 时，代码把 K 固定为 1，并按输入行数分配缓冲区；它不会从 q 推断原来的 Top-K。

### 4.2 index_copy_ 中的索引是写入位置

```python
unpermuted_tokens.index_copy_(
    0, torch.argsort(sorted_indices, stable=True), permuted_tokens
)
```

`argsort(q)` 重新得到 p。`index_copy_` 的语义是按目标位置写入，逻辑上等价于：

```python
for j in range(Y.size(0)):
    restored[p[j]] = Y[j]
```

| 专家输出的当前行 j | 目标槽位 p[j] | 写入的值 |
| --- | --- | --- |
| 0 | 1 | A0 |
| 1 | 4 | C0 |
| 2 | 2 | B1 |
| 3 | 5 | C1 |
| 4 | 0 | A2 |
| 5 | 3 | B2 |

恢复后的内容是 `[A2,A0,B1,B2,C0,C1]`。这一步恢复的是各 token 的路由槽位，还没有合并专家输出。

### 4.3 reshape、权重广播与求和

```python
unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
if probs is not None:
    unpermuted_tokens *= probs.unsqueeze(-1)
return unpermuted_tokens.sum(dim=1)
```

形状依次为 `[T*K,D] → [T,K,D] → [T,D]`。`probs.unsqueeze(-1)` 的形状是 `[T,K,1]`，给每一路专家输出的全部 D 个分量乘同一个权重。由于槽位顺序已经恢复，`probs` 应保持原 `[T,K]` 顺序，不需要再次按专家排序。

本例最终得到：

```text
A_out = 0.7*A2 + 0.3*A0
B_out = 0.4*B1 + 0.6*B2
C_out = 0.8*C0 + 0.2*C1
```

函数不会归一化 `probs`，也不会检查每行权重和是否为 1；它直接执行传入权重的加权和。`restore_shape=None` 且 `probs=None` 时形状变为 `[输入行数,1,D]`，最后的求和不会把原 K 路结果自动合成一个 token。

## 5. 三次 argsort：正确性与可以简化的部分

| 次数 | 位置 | 排序对象 | 结果与用途 |
| --- | --- | --- | --- |
| 1 | `eager_permute` | 展平专家编号 | p，决定专家分组与取 token 的顺序 |
| 2 | `eager_permute` | p | q，作为公共返回索引 |
| 3 | `eager_unpermute` | q | p，用于 `index_copy_` 写回 |

因此当前实现确实存在 `p → q → p` 的转换。生成 q 不等于恢复数据；只有实际执行 `Y[q]` 或 `restored[p]=Y`，数据才恢复顺序。

对于未截断的完整置换，保持 q 接口时可以直接用 `Y.index_select(0, q.long())` 恢复槽位，省去 unpermute 中的排序。若完全重新设计一对内部函数，也可以只保存 p，再按 p 写回。即使需要生成 q，也可以用 `q[p]=arange(N)` 构造逆置换，不必再次排序。

这些是等价变换的分析，不是本次对上游代码的修改或性能测量。公共 eager/fused 接口共享 q 的含义，不能只把 eager 的返回值换成 p。实际还有混用分支：[`ep_allgather_forward()`][allgather] 的 permute 传入 `fused=fused`，但末尾 unpermute 未显式传 `fused`，会使用默认值 True。

## 6. num_out_tokens 与 restore_shape 的实际边界

### 6.1 num_out_tokens 是对专家分组结果做切片

在当前 eager 实现中，`None` 和 0 都不截断；正数保留前若干行，超过 N 时保留全部；负数遵循 Python 切片语义。该参数本身不执行逐专家 capacity 分配。

例如 N=6、`num_out_tokens=4` 时，token 输出只有 4 行，但 q 仍长 6。不能直接把这对结果当作普通未截断的 unpermute 输入。

### 6.2 restore_shape 是 eager 合并前的缓冲区形状

```python
unpermuted_tokens = torch.zeros(restore_shape, dtype=..., device=...)
permuted_tokens = permuted_tokens[:restore_shape[0]]
```

设 `restore_shape=[R,D]`。源码先截取输入前 R 行，再把结果写入 `[R,D]` 缓冲区；有 K 路权重时，后续正常形状是 `[R/K,K,D]`，最终输出 `[R/K,D]`。

仓库[测试构造][test-unpermute]包含以下输入：

```text
permuted_tokens: [10,32]
sorted_indices: [8]，是 0～7 的排列
restore_shape:  [8,32]
probs:          [4,2]

[10,32] → 截前8行 → 恢复到[8,32] → [4,2,32] → [4,32]
```

所以 `[8,32]` 不是最终输出形状。`probs=None` 时 K=1，最终输出才仍为 `[8,32]`。

外层 `unpermute()` 仅在 `restore_shape is None` 时检查输入行数等于索引长度。传入该参数会跳过这一检查，但 `index_copy_` 仍要求索引数量与裁剪后的源行数一致，并满足目标范围与维度约束。源码不会同步裁剪索引。

例如，把前一节只有 4 行的输出和长度 6 的 q 传入 eager，并指定 `[6,D]`，仍会因为行数不匹配而失败。本次 CPU 检查复现了这一边界。不能因为缓冲区初始为零，就推断所有被丢弃的路由都会自动在正确位置补零。

### 6.3 dtype、连续性与数值假设

- 两次 eager permute 排序都先转为 float32。专家编号及位置过大时可能丢失整数精度；连续整数在 `2^24` 范围内可以精确表示。稳定排序不能修复 float 转换已经造成的数值合并。
- `indices.float().view(-1)` 要求转换后的张量满足 `view` 的布局条件；不要把它当成任意非连续输入都已验证的接口。
- eager 返回 q 的 dtype 与 `indices.dtype` 相同；参考融合接口返回 q 为 int32。两者索引含义相同，不代表 dtype 总是相同。
- eager 的概率乘法是原位操作，结果保持缓冲区 dtype；融合包装器则先把专家输出转为 `probs.dtype`。跨 dtype 情况不应直接假设数值与 dtype 完全一致。

## 7. NPU 融合算子：输入、输出与调用方式

本仓库的 [`fused_permute()`][fused-permute] 和 [`fused_unpermute()`][fused-unpermute] 是 `torch_npu` 接口的薄包装。Python 层把重排或恢复与合并交给单个扩展算子，而不再逐个调用 eager 的排序、取行、写回和求和操作。本文没有测量速度，也不推断融合内核内部执行了几次排序。

### 7.1 torch_npu.npu_moe_token_permute

本仓库调用方式：

```python
permuted_tokens, q = torch_npu.npu_moe_token_permute(
    tokens, indices, num_out_tokens=num_out_tokens
)
```

参考 [op-plugin 7.3.0 schema][op-schema] 的接口形态是 `npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False)`。当前 MM 包装器没有暴露或传入 `padded_mode`。

| 输入或输出 | 普通未填充模式的含义 |
| --- | --- |
| `tokens` | 输入 token；本文按 `[T,D]` 使用 |
| `indices` | `[T,K]` 或 `[T]` 的专家编号；参考接口支持 int32/int64 |
| `num_out_tokens` | 可选输出行数；普通未裁剪示例使用 None，0 表示不丢弃 |
| 返回 `permuted_tokens` | `[M,D]`，按专家分组的 token；未截断时 M=T*K |
| 返回 q | 长度 T*K 的逆置换；参考融合接口为 int32，表示原路由槽位在分组结果中的位置 |

融合 permute 的 token 输出 dtype 与输入 tokens 相同；这个约定与索引固定使用 int32 是两件事。

[CANN 8.5.0 的算子说明][cann-permute]明确给出 `sortedIndicesOut=argsort(argsort(indices))` 的索引约定，浮点 token 支持范围列出 FP16/BF16/FP32。设备支持、规模限制和参数组合应以实际部署版本为准。

### 7.2 torch_npu.npu_moe_token_unpermute

本仓库调用前会做 dtype 对齐：

```python
if probs is not None:
    permuted_tokens = permuted_tokens.to(probs.dtype)
return torch_npu.npu_moe_token_unpermute(
    permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
)
```

参考 [op-plugin 7.3.0 schema][op-schema] 的接口形态是 `npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None)`。

| 输入或输出 | 普通完整置换路径的含义 |
| --- | --- |
| `permuted_tokens` | `[T*K,D]`，专家计算后的输出，保持 permute 的分组行顺序 |
| `sorted_indices` | permute 返回的 q；参考融合接口要求 int32 |
| `probs` | 可选 `[T,K]` 路由权重，保持原顺序；None 时按 K=1 处理 |
| `restore_shape` | 本仓库直接透传；外部算子的具体约束与 eager 分支应分别核对 |
| 返回输出 | 有 probs 时 `[T,D]`；无 probs 时，普通完整置换路径仅恢复行顺序 |

经过本仓库包装器，有 probs 时输出使用 probs 对应的浮点 dtype；没有 probs 时保留输入专家输出的 dtype。

[CANN 8.5.0 unpermute 文档][cann-unpermute]描述的逻辑是按 q 读取专家输出，再按 probs 加权和归约。因此 q 是可用于 gather 的索引，并不要求在调用方先转回 p。

同一版外部文档把 `restore_shape` 与 `padded_mode=True` 联系起来，同时写明当前只支持 `padded_mode=False` 和空 restore shape。这个约束与仓库 eager 分支处理非空 restore shape 的方式不能混为一谈。本文未在 NPU 上验证非空 restore shape，也不保证某个安装版本支持裁剪后的恢复。

### 7.3 推荐从公共包装器理解调用边界

下面是调用结构示意，`run_local_experts` 表示调用方的专家计算，并非这两个算子提供的函数：

```python
import torch
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute

# NPU 示例：tokens、expert_ids、routing_weights 已在同一设备。
# tokens 和 routing_weights 使用配套支持的浮点类型。
expert_inputs, q = permute(
    tokens, expert_ids.to(torch.int32), num_out_tokens=None, fused=True
)
expert_outputs = run_local_experts(expert_inputs)  # 必须保持分组行顺序
output = unpermute(
    expert_outputs, q, probs=routing_weights, restore_shape=None, fused=True
)
```

仅切换这两个公共调用的 `fused=False` 可选择它们的 eager 分支，但不能据此断言整个模型没有其他融合算子。项目 `pyproject.toml` 声明 `torch==2.10.0`，并没有在该文件中锁定 `torch_npu`/CANN 的唯一配套版本；外部参考版本不等同于仓库运行环境。

## 8. 放回 Kimi K3 与 EP 调用链

Kimi K3 的 [`_forward_fused()`][kimi-fused]按以下顺序调用：

```text
permute → grouped_matmul → 激活 → grouped_matmul → unpermute(probs)
```

它显式把专家编号转为 int32，两个重排接口都传 `fused=True`。同一类的 `_forward_eager()` 是逐专家 mask/计算/累加的另一条实现，不应把“该模型的 eager 前向”直接等同于“本文两个公共算子的 eager fallback”。

普通 all-to-all EP 路径则有两层排列，见 [`alltoall_dispatch()`][alltoall-dispatch] 与 [`alltoall_combine()`][alltoall-combine]：

```text
按全局专家分组的 permute（K路）
  → all-to-all 分发
  → 按接收侧本地专家再 permute（该次索引为一维，K=1）
  → 专家计算
  → 本地 unpermute（probs=None）
  → all-to-all 返回
  → 原 token 的 unpermute（probs=routing_weights）
```

因此，在执行完整两层排列、且都选择 eager 的路径中，是两对算子、总共六次 `argsort`，而不是整个 EP 流程只有三次。通信由 dispatcher 调用，局部排序算子本身不创建通信组。

all-gather 路径另有本地专家筛选和 `num_out_tokens` 裁剪，并把 permute 前的形状传给 unpermute。该调用点的存在不能证明 eager 已正确支持所有裁剪场景，尤其需要结合第 6 节和外部融合版本约束判断。

## 9. 训练中的梯度与验证范围

这两个重排函数不持有可训练参数。eager 的取行、写回、乘法、reshape、求和通过 PyTorch autograd 传播梯度；专家参数属于前后的专家模块。

- permute 使一个 token 出现 K 次，反向时这些分支对原 token 的梯度累加。
- 对专家输出的梯度包含对应路由权重的缩放；可求导的 `probs` 也接收加权求和产生的梯度。
- 离散专家编号和排序索引没有通常意义上的可微梯度；路由器通过 scores/probs 的训练路径与离散选择需要区分。
- 重排函数不负责 EP/DP/FSDP 的梯度通信或优化器更新；这些职责由外围分布式模块承担。
- 参考 [op-plugin 7.3.0 的梯度注册][op-derivatives]，permute 对 tokens 的梯度交给 `npu_moe_token_permute_grad`，unpermute 对专家输出及 probs 的梯度交给 `npu_moe_token_unpermute_grad`。这是该外部版本的注册事实，本文的 CPU 检查不能替代实际安装版本的 NPU 前反向验证。

本次实际验证环境为 Windows、Python 3.13.13、PyTorch 2.7.1+cpu，使用 float64。由于不是项目声明的完整环境，通过 AST 从上述 commit 对应的当前源文件提取两个 eager 函数，直接执行原函数体，未导入或运行完整模型。检查内容如下：

| 检查 | 方法与结果 |
| --- | --- |
| 索引和 token 顺序 | 3-token、K=2 示例，p/q 关系与分组输出一致 |
| 加权前向 | 使用三个不同的 2×2 专家矩阵，与逐 token、逐 Top-K 直接计算的参考结果一致 |
| 梯度 | 对 token、probs、专家矩阵比较非均匀上游梯度下的结果，`rtol=atol=1e-12` 通过 |
| `probs=None` | 输出等于按 q 直接取行，不额外合并原 K 路槽位 |
| `restore_shape` | 输入 `[10,2]`、q 长度 8、probs `[4,2]`、恢复缓冲区 `[8,2]`，与直接参考结果一致 |
| `num_out_tokens` | 0 不截断；4 只裁剪 token 行，q 仍长 6；将这组不匹配输入交给 eager unpermute 复现 IndexError |

仓库自身的 [`test_permute.py`][test-permute] 和 [`test_unpermute.py`][test-unpermute]提供 eager/fused 测试构造，但本次未运行其 NPU 测试。尤其 `test_unpermute.py` 最后一段有 probs 和非空 restore shape 的分支仅调用两个实现，没有 `assert_close`，不能把该段代码存在当成数值一致性已经通过。

另外，公共入口在非 NPU 环境中会回退到 eager；即使在 CPU 上运行这些 `fused=True` 测试通过，也只能说明回退路径通过，不能证明 NPU 融合算子一致。

本文没有修改上游实现，没有测量耗时、显存或吞吐量。后续设备验证应优先覆盖普通完整置换的前反向对齐、概率 dtype、裁剪/恢复参数组合，以及真实 EP dispatcher 路径。

[返回 MindSpeed-MM 索引](README.md) · [返回仓库首页](../README.md)

[eager-permute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/permute.py#L8-L16
[fused-permute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/permute.py#L19-L20
[permute-entry]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/permute.py#L23-L27
[eager-unpermute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/unpermute.py#L8-L21
[fused-unpermute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/unpermute.py#L24-L27
[unpermute-entry]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/ops/moe_ops/unpermute.py#L30-L36
[kimi-fused]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/models/kimi_k3/kimi_moe_patch.py#L142-L166
[alltoall-dispatch]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py#L288-L316
[alltoall-combine]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py#L319-L344
[allgather]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py#L520-L581
[test-permute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/tests/ut_fsdp/ops/moe_ops/test_permute.py#L1-L30
[test-unpermute]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/tests/ut_fsdp/ops/moe_ops/test_unpermute.py#L1-L47
[op-schema]: https://gitcode.com/Ascend/op-plugin/blob/7.3.0/op_plugin/config/op_plugin_functions.yaml
[op-derivatives]: https://github.com/Ascend/op-plugin/blob/7.3.0/op_plugin/config/derivatives.yaml#L239-L244
[cann-permute]: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnMoeTokenPermute.md
[cann-unpermute]: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnMoeTokenUnpermute.md
