# MindSpeed-MM 模块异构并行：从通信组到训练边界

本文以 Qwen2.5-VL 路径为例，分析视觉编码器与 LLM 使用不同 TP、DP、PP、CP 配置时，MindSpeed-MM 如何初始化状态、搬运输入输出，以及复用视觉特征。

这里的“异构”指**模型模块的并行配置不同**，不指不同型号硬件，也不指插件式 FSDP2 后端。

| 项目 | 范围 |
|---|---|
| 源码仓库 | [Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| 分析版本 | `dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc` |
| 记录日期 | 2026-10-08 |
| 入口 | Megatron 训练路径 `pretrain_vlm.py` |
| 验证方式 | 本地源码静态分析，未执行分布式训练或数值对齐实验 |

下文链接固定到上述提交，结论不自动适用于其他版本；链接按源仓库路径生成，未验证远端页面可访问性。

## 1. 实现整体分成三个层次

1. **初始化**：按模块创建通信组，保存各自的并行状态；模型也按自己的配置构建。
2. **模块边界转换**：视觉前置 hook 将 LLM 布局的图像数据转换为视觉布局，后置 hook 将视觉特征转换回 LLM 布局。
3. **异构流水线调度**：先执行编码器阶段，再将保存的样本和特征交给 LLM；支持编码器一次处理多个 LLM microbatch。

```text
按 LLM DP 读取数据
       ↓
视觉前置 hook：LLM DP gather → 切换视觉状态 → 视觉 DP split
       ↓
image_encoder：ViT + projector
       ↓
视觉后置 hook：视觉 DP gather → 切换 LLM 状态 → LLM DP split
       ↓
视觉特征与文本 embedding 组合 → LLM
```

这条前向链路已经实现，不代表任意并行组合或 ViT、LLM 联合训练已经得到支持。反向限制见第 8 节。

## 2. 参数、初始化与状态切换

[参数定义][args]提供 `--hetero-parallel` 和 `--hetero-encoder-mbs-scale`，后者默认值为 1。

启用异构时，[参数校验][validate]要求命令行 TP、PP、CP 初始值均为 1，实际并行度来自模型配置：视觉模块使用外层 `image_encoder.tp/pp/cp`，文本模块使用 `text_decoder.tp/pp/cp`。[配置整理][configure]通过 `hetero_align_config()` 转为内部的长字段名，再建立模块配置。这里还包含 projector 的 CP 固定为 1 等模块约束。

参数校验会将全局 `args` 的 TP、PP、CP 更新为 LLM 配置，并重新计算 DP 和 `global_batch_size`。因此，命令行写入的 GBS 不一定等于最终生效的 GBS，应检查校验后的参数。

调用顺序是：

```text
initialize_megatron 的异构包装器
  → 原初始化
  → 复制并整理模块配置
  → initial_modules_mpu()
  → 构建各模块
  → apply_hetero_parallel_hooks()
```

入口见[初始化包装器][init-wrapper]，通信组创建见 [`initial_modules_mpu()`][init]。函数分别提取视觉、音频、文本配置，然后逐模块清理 `mpu` 当前登记的状态及 rank 缓存，调用 `initialize_model_parallel()`，保存快照：

```python
_ParallelStatesDict[module].update(state_snapshot)
```

参数选择优先使用模块配置。缺少字段时，文本模块或标记了 `use_args=True` 的参数从命令行参数对象读取；其他参数尝试 Megatron 配置类默认值，最后使用显式默认值。它不是所有字段都无条件依次尝试四层来源。

[`change_parallel_state()`][switch]将指定快照写回 `vars(mpu)`。`vars(mpu)` 是模块实际的属性字典，修改它就修改了模块全局变量。快照保存的是引用，包含通信组对象及缓存状态。

因此，状态切换是**更换当前查询所使用的状态**，不是重新创建通信组，也不会移动数据、切分权重或同步梯度。权重分片在模型构建时确定，数据转换由另外的通信和切分代码完成。

## 3. 为什么 DP 组能够收集到完整样本集合

以 4 张卡、LLM `TP=2、DP=2、PP=CP=1` 为例，采用下面的 rank 布局：

| DP 副本 | TP 位置 0 | TP 位置 1 | 当前输入 |
|---|---|---|---|
| 0 | rank0 | rank1 | 样本 0～3 |
| 1 | rank2 | rank3 | 样本 4～7 |

- TP 组为 `[0,1]`、`[2,3]`：协作计算相同样本的不同模型分片。
- DP 组为 `[0,2]`、`[1,3]`：持有相同模型分片，处理不同样本。

每个 DP 组都覆盖两个 DP 副本，因而可以分别收集到样本 0～7 的视觉输入。它不需要包含全部 4 张卡。这里依赖同一 TP 副本内的视觉原始输入一致，并不表示任意 PP、CP 配置都具有相同的数据布局。

“完整”限定为**当前 microbatch 跨 DP 的样本集合**，不是整个数据集，也不是包含所有梯度累积步骤的全局 batch。

## 4. 数据从哪里来，为什么最初是 LLM 布局

[`train_valid_test_datasets_provider()`][data-provider]先切换到 `text_decoder` 状态，再把其 DP 组传给 DataLoader。[分布式 sampler][sampler]使用该组的大小和组内 rank 确定数据份数及本地分片。

当 `hetero_encoder_mbs_scale=k>1` 时，构建 DataLoader 前临时将 `args.micro_batch_size` 乘以 `k`，[DataLoader 读取此值作为 batch size][loader]。构建完成后恢复 `args.micro_batch_size`，但已建立的 DataLoader 仍按放大后的批量读取。

这时读取批量仍以 **LLM DP 分片**为单位，视觉本地批量需要经过前置 hook 才能确定。启用 `use_data_balance` 还有额外的大批量读取和重排逻辑，不能把普通分支的读取过程直接套用过去。

## 5. 视觉前置 hook：按样本关系重新分配图片

[`image_encoder_forward_pre_hook()`][pre-hook]接收三个输入：

| 输入 | 含义 | 第 0 维 |
|---|---|---|
| `pixel_values` | 经过图像预处理、按 patch 展平排列的像素数据 | patch 数量 |
| `image_grid_thw` | 每张图像的 `[T,H,W]` patch 网格 | 图片数量 |
| `text_img_num` | 每条文本样本对应的图片数量 | 样本数量 |

`pixel_values` 还不是压缩后的语义特征。以 RGB、空间 patch 为 `14×14`、时间 patch 为 2 为例，一行可以包含 `3×2×14×14=1176` 个数值；之后的 patch embedding 才将其映射到模型隐藏维度。`H/W` 是预处理后尺寸除以 patch 大小得到的网格数量。

外层 [`VLMModel.forward()`][vlm-forward]通过视觉起始 token 计数生成 `text_img_num`，再调用已注册 hook 的视觉模块。

前置 hook 的顺序为：

1. 切换到 LLM 状态，分别 gather 三个输入。
2. 切换到视觉状态，查询视觉 DP 数量。
3. 用 `torch.chunk(text_img_num, 视觉DP数)` 按样本分块。
4. 对每块求和，得到每个视觉 DP rank 的图片数量。
5. 根据图片网格计算 patch 总数，切分两个图像张量。
6. 返回 `pixel_values, image_grid_thw`，作为视觉 `forward()` 的输入。

例如，视觉 DP=2，四条样本的图片数是 `[1,2,1,0]`：

```text
样本分配：[1,2] | [1,0]
图片数量：   3  |  1
图片顺序： A B C | D
patch数： 4 8 16 | 4
patch合计：  28  | 4
```

于是 `thw_num_per_DP_rank=[3,1]`、`pv_lens=[28,4]`。视觉 DP rank 0 取前 3 行网格及前 28 行像素 patch；rank 1 取后 1 行网格及后 4 行像素 patch。

这里均分的是样本，不保证图片数、分辨率或实际计算量均衡。

### 变长 gather 与本地 split

[`all_gather_dp_group()`][gather]先 gather 长度，例如得到 `[28,4]`，将两边都补成 28 行，再 gather 内容、按原长度去掉 padding，最后拼成 32 行。当前实现用 `zeros_like()` 建立等形状接收缓冲区，所以需要这一步补齐；不应据此断言所有 all-gather API 都只支持等长输入。

视觉调用使用 `pad_dim=0`，与函数内部 `g[:length]` 的裁剪方式一致。其他维度仍需匹配，不能把该实现当作任意维度变长 gather。

[`split_tensor_dp_group()`][split]不发送数据。它根据当前 DP 组内 rank，在本地执行：

```python
torch.split(tensor, chunk_seq_lens, dim=split_dim)[rank]
```

指定长度列表时，其总和必须等于切分维度的长度；未指定时，函数改用 `torch.chunk()`。输入过小或不能均分时，也要关注实际块数和分配是否满足调用方假设。

## 6. 视觉后置 hook：把特征交回 LLM

[`image_encoder_forward_hook()`][post-hook]在视觉模块完成后执行，顺序是：视觉 DP gather 输出及各段长度 → 切换 LLM 状态 → 合并长度 → 按 LLM DP rank 取出特征。

例如视觉 DP=4、LLM DP=2，四份视觉输出长度为：

```text
[6,10,8,12] → [6+10, 8+12] → [16,20]
```

每两个相邻视觉 DP rank 的输出对应一个 LLM DP 副本。这里要求视觉 DP 数量不小于 LLM DP，且是它的整数倍，还依赖连续分组后的样本顺序与 LLM 原分配一致。

长度对应的是 **ViT 和 projector 之后的特征行数**，不是输入 patch 数。空间 `2×2` 合并等操作可能减少特征行数。[视觉模块 forward][vision-forward]同时包含编码器和 projector，这一点也影响冻结与反向边界。

## 7. `hetero_encoder_mbs_scale` 与批次重放

令 LLM 本地 MBS 为 `B`、scale 为 `k`。样本可均分且布局满足上述假设时：

```text
视觉本地样本数 = B × k × LLM_DP / 视觉_DP
```

不同 DP 数量改变本地批量；`k` 进一步将多个 LLM microbatch 合并到编码器阶段。例如 `B=2、k=4、LLM_DP=2、视觉_DP=4`，视觉每个 DP 副本一次处理 4 条样本。

专用调度器由 [`train_step()`][train-step]在 `hetero_parallel` 且 LLM `PP>1` 时选择。因此，普通 `PP=1` 路径应按 `k=1` 使用和分析，不能只放大 DataLoader 就认为批次重放已经生效。

[`hetero_pipeline()`][hetero-pipeline]将编码器 microbatch 数改为 `N//k`，LLM 阶段恢复为 `N`。需要 `k>0` 且 `N` 能被 `k` 整除，源码中的整数除法不代表非整除配置也正确。scale 不以增加一次优化器更新的总样本数为目的。

### 两个迭代器分别做什么

[`ReplayIterator`][replay]包装真实迭代器，每次 `next()` 读取一个大 batch，并在 `_current_batch` 保存引用。它只记住最新一批；多批历史由调度器另行收集为 batch 列表和输出列表。

编码器阶段的模型提前返回 `[vit_embeds, audio_embeds]`。进入 LLM 阶段后，[`DecoderRerunDataIterator`][rerun]将保存的 batch 和输出配对，按 `k` 拆小并逐次 `yield`。模型读取其中的 `vit_embedings` 字段，跳过视觉计算，直接使用缓存特征；字段拼写以源码为准。

假设 DP 相同、每条样本一张图，`B=2、k=2`，一次迭代有 4 个 LLM microbatch：

```text
编码器第1次：[样本0、1、2、3] → 缓存特征V0
编码器第2次：[样本4、5、6、7] → 缓存特征V1

LLM第1次：样本0、1 + V0对应片段
LLM第2次：样本2、3 + V0对应片段
LLM第3次：样本4、5 + V1对应片段
LLM第4次：样本6、7 + V1对应片段
```

这是 4 个 LLM microbatch 的前向，不是 4 次优化器更新。

### 重放实现的具体假设

迭代器通过网格乘积的累计和、再除以写死的 `VIT_SCALE_FACTOR=4`，定位视觉特征。例如输入 patch 数为 `[4,8,16,4]`，输出边界计算为 `[0,1,3,7,8]`。前两张图取特征 `[0:3]`，后两张图取 `[3:8]`，不能直接均分特征行。

此外，代码以图片数量计算 `start_idx/end_idx`，同时用它切文本等所有 Tensor 字段，没有利用 `text_img_num` 处理一般的多图、无图样本边界。因此需要图片与样本索引对应的约束，不能从前置 hook 支持按图片计数推断重放器也支持任意样本结构。

所有 Tensor 都沿第 0 维切片，对 `pixel_values` 并不是正确的 patch 边界；当前 LLM 阶段跳过视觉前向，使用单独切好的缓存特征。音频路径还存在固定 token ID 的约定。

## 8. 反向传播：当前不能据此联合训练 ViT 与 LLM

这套实现没有注册与两个 forward hook 对称的 backward hook。两个直接限制是：

1. 视觉后置 hook 使用 `remove_padding=True`；[`all_gather_dp_group()`][grad-limit]遇到 `tensor.requires_grad=True` 且需要去 padding 时，直接抛出 `NotImplementedError`。错误发生在前向收集输出时。
2. [异构 PP 外层调度][hetero-pipeline]对非最后模块阶段设置 `forward_only=True`，没有完整的反向模块遍历。这里“最后模块”指 LLM，不是只有 LLM 的最后一个流水线 rank 做反向。

因此，应将当前路径理解为**冻结视觉特征提取部分，用特征训练 LLM**，不能声称已支持 ViT 与 LLM 端到端联合训练。仅冻结 ViT 而继续训练 projector，视觉输出仍可能需要梯度，也会遇到第一项限制。[冻结默认值][freezing]中 ViT 与 projector 不同，不能只检查 ViT。

另有自定义 [`_AllGatherDp.backward()`][gather-backward]，但它只按前向保存的组内 rank 和输入长度切出本地梯度，没有跨 rank 汇总梯度通信，不是任意布局下通用的可微 all-gather。

状态切换也不意味着已经创建的 DDP 对象会自动更换自己保存的通信组。LLM 的常规反向、DP 梯度同步和优化器更新，需要与模块间特征转换分开理解。

## 9. 后续阅读与验证方向

- [模块 CP attention][cp]：按模块配置选择 attention，并通过 all-to-all 转换序列与 head 布局；它与模块边界的 DP gather/split 是不同层面。
- [数据均衡异构分支][balance]：观察视觉 DP 数量如何参与负载分配，避免将样本数均分等同于 patch 计算量均衡。
- 音频前后 hook 与视觉采用相似框架，但 padding、输入结构和重放约定不同，需单独检查。
- 真正运行前，应核对 DP 比例、样本数可分性、图片与文本对应顺序、scale 整除性、冻结设置和特征输出类型。源码存在分支不等于该配置已通过分布式或精度验证。

[返回 MindSpeed-MM 索引](README.md) · [返回仓库首页](../README.md)

[args]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/arguments.py#L114
[validate]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/patchs/validate_args_patch.py#L357
[configure]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/pretrain_vlm.py#L72
[init-wrapper]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L196
[init]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L107
[switch]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L187
[data-provider]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/pretrain_vlm.py#L195
[sampler]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/data/dataloader/dataloader.py#L219
[loader]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/data/__init__.py#L128
[pre-hook]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L32
[vlm-forward]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/models/vlm_model.py#L600
[gather]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L219
[split]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L272
[post-hook]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L56
[vision-forward]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/models/vision/vision_model.py#L115
[train-step]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/training.py#L751
[hetero-pipeline]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/patchs/hetero_pipeline_patches.py#L246
[replay]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/patchs/hetero_pipeline_patches.py#L50
[rerun]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/patchs/hetero_pipeline_patches.py#L75
[grad-limit]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L255
[freezing]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/pretrain_vlm.py#L103
[gather-backward]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_parallel.py#L337
[cp]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/hetero_utils/hetero_CP_utils.py#L20
[balance]: https://gitcode.com/Ascend/MindSpeed-MM/blob/dd6297017f4c967d8c32cd9f6cd0ea2c65e370cc/mindspeed_mm/utils/data_balance/data_balance.py#L145
