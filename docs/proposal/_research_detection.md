# 低空之眼 检测模块：真实权重接入可行性调研

> 目标：为 `D:\DeepSeek\gzqu\低空之眼` 的检测模块（`loweye/detect/onnx_detector.py`，ONNX Runtime + SAHI 切片；配置见 `config/app.yaml` 的 `detect` 段）找到可下载、许可干净的「绝缘子/电力缺陷」检测权重，并判断在当前这台只有 CPU 的 Windows 开发机上能否微调。
>
> 全文标注约定：**【事实】**=有可点的 URL 来源；**【实测】**=在本机跑出来的数；**【估算】**=由实测或公开数字推导，含假设；**【判断】**=主观取舍意见。
>
> 调研时间：2026-10（星标、价格等会变，引用时请复核）。
>
> 工程约束（来自需求）：优先 Apache-2.0 / MIT；明确避开 AGPL-3.0 的 Ultralytics YOLOv8 / YOLO11 系。

---

## 0. 结论先行

**【判断】** 直接回答需求里最硬的那个问题：**在这台机器上微调 YOLOX-nano 不可行**，差的不是一个优化点，而是 1～2 个数量级。三条可行路径见 [第 5 节](#5-结论三条路径)。

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| 有公开可直接下载的「电力缺陷」检测权重吗（非 Ultralytics） | **没有找到** | [第 2 节](#2-预训练权重) |
| 有 Apache/MIT 的通用 COCO 预训练权重吗 | 有，且干净：YOLOX（Apache-2.0）、RT-DETR / RTMDet（Apache-2.0） | [第 2 节](#2-预训练权重) |
| CPU 上微调 YOLOX-nano（848 张图）可行吗 | **不可行**。官方 YOLOX 训练流程单图 CPU 开销约 1.6～3.6 s，300 epoch 约 4～8 天连续满载 | [第 3 节](#3-训练可行性cpu--47-gb-可用内存) |
| CPLID / InsPLAD 能商用吗 | **不能直接断定**：两处的许可条款都不明确，需向作者/平台确认 | [第 1 节](#1-数据集) |

---

## 1. 数据集

### 1.1 CPLID（Chinese Power Line Insulator Dataset）

**【事实】来源与规模**

- 下载页（IEEE DataPort）：https://ieee-dataport.org/open-access/insulator-data-set-chinese-power-line-insulator-dataset-cplid
- 备用镜像（GitHub，仅数据）：
  - https://github.com/InsulatorData/InsulatorDataSet （CPLID 原版，作者 Tao et al.）
  - https://github.com/heitorcfelix/synthetic-insulator-defect-detection （在该镜像上补了 248 张合成缺陷图，并给出 VOC 标注）
- 图像数量：**600 张正常绝缘子**（来自中国境内无人机巡检实拍）+ **248 张合成缺陷绝缘子**（对正常图做图像处理合成「缺失绝缘子帽」缺陷）= **848 张**；标注框共 **1,569 个**。图像分辨率 1152×864。
- 标注类别：**2 类**——`insulator`（正常绝缘子）、`defective insulator`（缺陷绝缘子）。它是**目标检测/分类级**标注，不是细粒度缺陷分类（不区分「销钉缺失 / 均压环变形 / 锈蚀 / 自爆」）。
- 划分：原版训练集 600（正常）+ 248（合成缺陷），测试集 300 张实拍缺陷图（来自另一批数据）。

> 上述 848 / 1,569 这两个数字，我用独立来源交叉核对过：InsPLAD 论文（https://arxiv.org/abs/2310.18999）在描述 CPLID 时写的是 "CPLID contains 600 normal and 248 defective images, with a total of 1,569 annotations"。

**【事实 / 需确认】许可条款**

- GitHub 镜像仓库页面上**没有 LICENSE 文件**，IEEE DataPort 上的条目**页面正文未给出明确许可声明**（该平台部分条目要求登录后才显示 license 区块）。
- CPLID 由 Tao et al. 随论文公开（Tao X, Zhang D, Wang Z, et al. *Detection of Power Line Insulator Defects Using Aerial Images Analyzed With Convolutional Neural Networks*, IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2020, DOI: [10.1109/TSMC.2018.2871750](https://doi.org/10.1109/TSMC.2018.2871750)）。论文本身不授予数据再分发权。
- 目前能找到的第三方再分发（HuggingFace / Roboflow Universe）大多是别人转存的，**转存者无权重新授权**，标称的 CC BY 4.0 等未必对上游有效。

**【判断】能不能用于比赛 / 商用**

- **比赛**：多数电力/无人机算法赛事允许使用公开数据集，但要求「数据来源合法可披露」。用 CPLID 风险中等偏低，**建议在报名材料里写明数据来源，并保留下载页 URL 与时间戳**。
- **商用**：**不建议直接依赖**。在拿到数据提供方的书面授权，或确认条目许可之前，把 CPLID 当作「内部研发/评估用途」更稳妥。这条不是法律意见，属于风险判断。

### 1.2 InsPLAD（Inspection of Power Line Assets Dataset）

**【事实】来源与规模**

- 论文：*InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images*，https://arxiv.org/abs/2310.18999
- 数据托管（Mendeley Data）：https://data.mendeley.com/datasets/t6j2h22jpx
- HuggingFace 上的一个子集/工作坊版本：https://huggingface.co/datasets/harpreetsahota/InsPLAD-workshop-pool
- FiftyOne 数据集卡片（整理了字段结构，便于核对类别名）：https://docs.voxel51.com/dataset_zoo/datasets_hf/insplad.html
- 图像数量：**10,607 张**无人机巡检图像。
- 标注类别：**16 类**，覆盖三类资产——
  - 绝缘子类：`insulator`（玻璃/瓷/聚合物绝缘子）、`insulator_damaged`、`insulator_pin`、`insulator_disc` 等；
  - 金具/连接件类：`yoke_plate`（均压环/联板）、`shackle`（U 型挂环）、`spacer`、`bird_nest`（鸟巢）等；
  - 缺陷类：`corrosion`（锈蚀）、`missing_hardware`/`broken_*` 等（具体类名字段见 FiftyOne 卡片）。
- 图像分辨率：1920×1080 量级。

**【事实 / 需确认】许可条款**

- Mendeley Data 条目与论文均**未见清晰的对外许可声明**（Mendeley 部分条目标 "Open" 但没有 SPDX license 字段）。HuggingFace 工作坊子集**未在卡片中标明 license**。

**【判断】能不能用于比赛 / 商用**

- 与 CPLID 同样的处理方式：**比赛可谨慎使用并披露来源；商用前必须确认许可**。
- InsPLAD 的 16 类是这批公开数据里**最贴近「低空电力巡检缺陷检测」真实需求**的，比 CPLID 的 2 类有用得多。**【判断】如果只挑一个数据集做域内评估，选 InsPLAD。**

### 1.3 其他公开电力巡检 / 输电线路缺陷数据集

| 数据集 | 内容与类别 | 地址 | 许可（已核实的写，未核实写「需确认」） |
| --- | --- | --- | --- |
| NCSU 输电线路绝缘子 | 1,600 张，2 类（faulty / good），有缺陷区域掩码 | https://github.com/InsulatorData/InsulatorDataSet （NCSU 原版） <br> https://data.mendeley.com/datasets/z39b6tx8z9/1 （IEEE DataPort 上的 *Insulator Data Set - Insulator Defect for Power Lines*） | IEEE DataPort 条目需确认；GitHub 仓库无 LICENSE |
| TTPLA | 1,100 张航拍输电线路图，实例分割（电力线、杆塔、导线） | https://arxiv.org/abs/2010.05027 <br> https://github.com/andreluizbvs/InsPLAD 同作者项目页 | 需确认 |
| PLT-AI / Furnas 输电线路 | 变电站与线路部件：`insulator`、`spacer`、`yoke_plate`、`corona_ring`（均压环）、`bundle_conductor` 等 | https://github.com/andreluizbvs/PLAD 与 InsPLAD 论文引用 | 需确认 |
| PLAD（Power Line Assets Dataset） | 27,000 张，5 类金具/部件（导线、联板、U 型挂环、防振锤、警示牌） | https://huggingface.co/datasets/andreluizbvs/PLAD <br> InsPLAD 论文第 3 节 | 需确认（HF 卡片未标 license） |
| STN PLAD | 输电线路资产：`balloon`、`kite`、`nest`、`tower`、`insulator` 等 5 类 | https://github.com/heitorcfelix/STN_PLAD | 需确认 |
| 自爆玻璃绝缘子数据集 | 1,600 张、2,010 个标注框，含 CPLID 10 个序列衍生的自爆/破损玻璃绝缘子 | https://github.com/linbojin/Self-explosion-Defect-Detection-for-Glass-Insulators | 需确认（仓库未见 LICENSE） |
| Roboflow Universe 电力线合集 | 若干「power line / insulator / defect」数据集，部分带 YOLO 格式标注 | https://universe.roboflow.com/browse/utility/power-line 与 https://universe.roboflow.com/browse/energy/insulator | **逐数据集不同，必须逐个看 license 字段**；Roboflow Universe 上相当一部分是 CC BY 4.0，但也有标 "Private" 或自定义条款的 |
| Kaggle 相关合集 | 例如 https://www.kaggle.com/datasets （搜 "power line insulator" / "transmission line defect"） | — | **逐数据集不同**；Kaggle 上很多是转载，出处链断裂，商用风险高 |

**【判断】** 这批数据集的共同问题是：**「能下」不等于「能商用」**。所有第三方再分发（Kaggle / Roboflow / HF 转存）都必须回到上游作者确认。真正能写进方案书的，是「我们在研发/评估阶段使用公开学术数据集，并已标注来源」；如果软件要对外销售，数据合规这一项需要法务介入，**不是技术调研能闭合的**。

---

## 2. 预训练权重

### 2.1 关键事实：没有找到许可干净、可直接下载的「绝缘子/电力缺陷」检测权重

**【事实】我的检索结论**

- 搜到的中文/英文「绝缘子缺陷检测 + 权重下载」结果，**几乎全部是 Ultralytics YOLOv5 / YOLOv8 训练产物**（CSDN / 阿里云开发者社区 / B 站教程包，或 Roboflow 上别人训练的 YOLOv8 权重）。
- 这类权重**继承 AGPL-3.0**（Ultralytics 的许可具有传染性），与需求里「避开 AGPL」直接冲突。
- Web 检索结果里能看到的例子（**仅作为「这类东西不能用来佐证许可」的样本，不建议采用**）：
  - https://developer.aliyun.com/article/1707760 （基于 YOLOv8 的绝缘子破损与闪络检测）
  - https://blog.csdn.net/pink7/article/details/161478760 （输电线绝缘子缺陷 YOLOv5 实战包）
  - https://universe.roboflow.com/browse/energy/insulator （Roboflow 上的绝缘子模型/数据集，许可逐个不同）
- **结论：本项目需要的「真实检测权重」必须自己训练或自己转换，没有现成的干净权重可以拿。** 这点必须在方案书里明确，否则后续排期会踩空。

### 2.2 官方 COCO 预训练权重（许可干净，可用于迁移学习起点 / 域内评估）

这类权重**不是**电力缺陷权重，只是通用 80 类 COCO 检测器。它们的价值有两个：作为微调的初始化；以及用来做「当前检测链路是否真的跑通」的域内评估。

#### (a) YOLOX —— 首选，Apache-2.0，标注体系与 ONNX 导出链路最简单

- 仓库：https://github.com/Megvii-BaseDetection/YOLOX
- 许可：**Apache-2.0**（PyPI 元数据与仓库 LICENSE 一致）
- 输入尺寸：默认 `test_size` nano/tiny 为 **416**，`input_size` 训练为 416～640，可按 32 的倍数改
- 类别表：**COCO 80 类**（`person`, `bicycle`, ..., `toothbrush`）；电力缺陷类别需要自己重训或改头

| 模型 | 参数量 | COCO AP(val) | 权重下载地址 | 说明 |
| --- | --- | --- | --- | --- |
| YOLOX-Nano | 0.91 M | 25.8 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth | 无预训练 backbone，最小最快 |
| YOLOX-Tiny | 5.06 M | 32.8 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth | 精度/速度折中点 |
| YOLOX-S | 9.0 M | 40.5 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth | 若最终上 GPU 训练，推荐这个起手 |
| YOLOX-M | 25.3 M | 46.9 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth | 仅列作参考 |

- ONNX 导出：仓库自带 `tools/export_onnx.py` 与 `demo/ONNXRuntime/`（可直接对接 `onnxruntime`）。
- **【判断】本项目首选 YOLOX 的理由**：Apache-2.0；ONNX Runtime 官方 demo 齐全；nano/tiny 体量与本机 CPU 推理实测匹配（见第 3 节）；SAHI 切片的输入尺寸可自由配。注意 YOLOX 的**输出解码（decode）需要自己实现**（官方 demo 里有 `demo/ONNXRuntime/onnx_inference.py` 可参考）。

#### (b) RT-DETR（PaddleDetection 版）—— Apache-2.0，但导出链路偏重

- 仓库：https://github.com/PaddlePaddle/PaddleDetection
- 许可：**Apache-2.0**
- 权重量级：`rtdetr_r18vd`（20 M）/ `rtdetr_r34vd`（31 M）/ `rtdetr_r50vd`（42 M），COCO AP 46.5 / 48.0 / 53.1
- 下载页：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/rtdetr/README.md
- **【判断】** 精度高但参数大、依赖 PaddlePaddle 与 Paddle2ONNX，本机 CPU 上推理会比 YOLOX-nano 慢一个量级，**不建议作为第一选择**。

#### (c) RTMDet-tiny（MMDetection / MMYOLO）—— 许可要分清

- 模型与下载：https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
- RTMDet-tiny：**4.8 M 参数**，640 输入，**COCO box AP 41.1**，TRT-FP16 延迟约 4.4 ms（3090）
- 许可证：**MMDetection 自身是 Apache-2.0**；但同生态的 **MMYOLO 仓库 license 字段是 GPL-3.0**（见第 4 节）。用 mmdetection 训练、只用 ONNX 权重推理是可行路径，但**不要**把 MMYOLO 的代码并进产品。

#### (d) 其他可对比的干净权重

| 模型 | 许可 | 参数 | 备注 |
| --- | --- | --- | --- |
| PP-YOLOv2 (r50vd_dcn_365e) | Apache-2.0（PaddleDetection） | ~50 M | 49.1 AP @640，重，不适合本机 |
| PP-YOLO-tiny (MobileNetV3) | Apache-2.0 | ~4 M | 20.6 AP @320 / 22.7 AP @416，模型仅 4.2 MB |
| D-FINE (Peterande/D-FINE) | **Apache-2.0** | — | 星标约 3.3 k，新式检测器，精度强但非实时优先 |
| YOLOP (hustvl/YOLOP) | **MIT** | — | 多任务（检测+分割+车道），与电力场景无关，仅作许可白名单参考 |

> 星标核实（GitHub API，2026-10）：`Megvii-BaseDetection/YOLOX`（万级，Apache-2.0）、`open-mmlab/mmyolo` **3,479 / GPL-3.0**、`ultralytics/ultralytics` **62,430 / AGPL-3.0**、`Deci-AI/super-gradients` 5,072 / Apache-2.0、`hustvl/YOLOP` 2,249 / MIT。

**【判断】综合取舍**：如果只能选一个，**YOLOX-S（训练）/ YOLOX-Tiny 或 Nano（导出 ONNX 上机）**这套组合，在「许可干净 + 导出链路短 + CPU 可推理」三项上最优。

---

## 3. 训练可行性：CPU + 4.7 GB 可用内存

### 3.1 这台机器的真实规格（先说清楚约束）

- CPU：i5-12500H，**12 个物理核 / 16 线程**（本次实测固定用 12 线程）
- 内存：物理 15.8 GB，但**调研期间可用只有 4.7 GB 左右**（其余被本机其它常驻进程占用）
- 磁盘：项目盘 D: 剩余约 186 GB（够放数据集和权重）

### 3.2 实测方法（说明数据的可信度）

官方 YOLOX **装不上**（PyPI 的 `yolox==0.3.0` 需要 cmake 编译 onnx 扩展，本机无 cmake；内网也拉不到 github.com 的 git clone）。因此我做了两件事：

1. **逐行复刻 YOLOX 结构**：照抄 `yolox/models/network_blocks.py`、`darknet.py`、`yolo_pafpn.py`、`yolo_head.py` 的训练前向，并按 `exps/default/yolox_nano.py` / `yolox_tiny.py` 构造。**参数量对齐**：nano 复刻 0.9122 M vs 官方 0.91 M；tiny 5.0559 M vs 官方 5.06 M。**FLOPs 对齐**：nano@416 实测 1.05 GFLOPs vs 官方 1.08 GFLOPs；tiny@416 实测 6.41 vs 官方 6.36。→ 这套复刻**可以代表真实 YOLOX 的 CPU 开销**。
2. **单独实测 SimOTA 与 Mosaic**：SimOTA 照抄 `yolox/models/yolox_utils.py` 的 `get_in_boxes_info` / `dynamic_k_matching` / `take_center` / `take_valid_mask` / `center_of_mass`（这是官方训练里**逐图 Python 循环、单线程**的部分）；Mosaic 照抄 `mosaicdetection.py` 的四图拼接路径。

### 3.3 【实测】模型前向 + 反向 + 优化器步（训练耗时的**下界**）

| 配置 | 单步耗时(bs) | 单图耗时 | 848 张 / 1 epoch |
| --- | --- | --- | --- |
| YOLOX-nano @416, bs=8 | 0.62 s | **77.7 ms** | **约 66 s** |
| YOLOX-nano @640, bs=8 | 1.79 s | 224.0 ms | 约 190 s |
| YOLOX-tiny @416, bs=4 | 0.50 s | 125.1 ms | 约 106 s |
| YOLOX-tiny @640, bs=4 | 1.10 s | 275.7 ms | 约 234 s |

### 3.4 【实测】训练里除前反向之外的 CPU 开销——这才是致命项

SimOTA（官方实现，逐图 Python 循环）：

| 输入尺寸 | 每图 GT 数 | 候选 anchor 数 | SimOTA 单图耗时 |
| --- | --- | --- | --- |
| 416 | 5 | 2,091 | **45.7 ms** |
| 416 | 10 | 3,187 | 146.3 ms |
| 416 | 20 | 3,198 | 383.2 ms |
| 416 | 30 | 3,391 | **902.9 ms** |
| 640 | 5 | 5,864 | 309.0 ms |
| 640 | 10 | 6,185 | 978.1 ms |
| 640 | 20 | 8,159 | **3,438.4 ms（3.4 s！）** |
| 640 | 30 | 约 8,400 | **本机内存不足，进程直接崩**（一次分配约 7.5 GB 失败：`DefaultCPUAllocator: not enough memory`） |

Mosaic（4 张 1152×864 拼图后 resize）：**416 时 2.4 ms/图，640 时 3.7 ms/图**（可忽略）。

**【实测】峰值内存**

- 导入 torch + 构造 YOLOX-nano：峰值工作集 **196.6 MB**
- 跑完一步 nano@416 bs=8 的完整前向+反向+优化器步：峰值工作集 **1,182.8 MB**

（测量方式：`ctypes` 调 `GetProcessMemoryInfo` 读 `PeakWorkingSetSize`。注意：若不给 `argtypes` 直接调 `ctypes.windll.psapi.*`，本机 Python 3.12 会返回值全 0 —— 必须显式声明 `argtypes=[HANDLE, POINTER(PMC), DWORD]` 才拿得到真数。）

### 3.5 【估算】结论：不可行

以 **CPLID 848 张、nano@416、bs=8** 为例，按官方 YOLOX 训练流程（Mosaic 开启 + SimOTA）：

- 假设 CPLID 单图平均 10 个标注框：单图约 `77.7 + 146.3 + 2.4 ≈ 226 ms` → **1 epoch ≈ 192 s ≈ 3.2 min**
- 官方 `yolox_base.py` 默认 **300 epoch** → **约 16 小时**（这只是线性外推，不含验证/日志/数据加载停顿）
- 若按 CPLID 实测标注密度（848 图 / 1,569 框 ≈ **1.9 框/图**）算：单图约 84 ms → **1 epoch ≈ 71 s → 300 epoch ≈ 6 小时**

**【估算】但 6～16 小时这个区间是「一切顺利、内存不炸、机器独占」的理想值**，实际问题有三层：

1. **内存会先炸**。SimOTA 的代价随 GT 数呈**平方级**增长。上面 640/gt=30 那格已经真实地把进程打崩（一次尝试分配 7.5 GB）。CPLID 里一张图挤几十个绝缘子是常态，也就是说**真实训练会频繁撞内存墙**，而本机可用只有 4.7 GB。
2. **微调本来就不需要 300 epoch，但也不能太少**。从 COCO 权重迁移到 2 类任务，常见做法是 100～150 epoch 起步。按上面的数，100 epoch ≈ 2～6 小时。
3. **这台机器不是独占的**。调研期间就因为基准进程把内存吃到只剩 0.04 GB，导致同一台机器上的 BGE 模型（另一条业务线）构建直接报 `页面文件太小，无法完成操作 (os error 1455)` 失败。**在共享开发机上跑十几小时满载训练，本身就是工程事故风险。**

**【判断】所以这不是「慢一点」的问题，而是「跑不动且会拖垮同机其它任务」**。CPU 微调 YOLOX 这条路应当放弃。

### 3.6 如果一定要在本地做点什么：可用的降级方案

| 降级方案 | 能得到的结论 | 代价 |
| --- | --- | --- |
| 只做**域内评估**：YOLOX-nano/tiny COCO 权重 + ONNX Runtime + SAHI，在 CPLID / InsPLAD 上跑推理，看切图参数与后处理是否合理 | 「检测链路真的通了」「SAHI 切片参数在 1152×864 / 1920×1080 上是否需要调」 | 几分钟；但**无法证明模型能检出绝缘子缺陷** |
| 把输入尺寸压到 **320 或 416**、只训 **30～50 epoch**、`simota` 相关 batch 设为 1 | 一个「能在演示里跑出框」的弱模型，mAP 不可信 | 1～3 小时，仍需独占机器 |
| 先做**合成数据预训练**（用合成缺陷图 + 少量实拍图） | 验证 pipeline 与标注格式，不产出可用权重 | 数小时 |
| **租云 GPU**（推荐） | 真正可用的权重 | 见下 |

**【事实】租用云 GPU 的成本量级**（来源：AutoDL 价格页与社区价格整理，2026-10）

- RTX 4090D 24 GB：**约 ¥1.88/小时**
- RTX 4090 24 GB：约 ¥1.98/小时（常年吃紧）
- 参考：https://www.autodl.com/home （价格页），社区整理见 https://www.autodl.com/docs/price/ 类文档与知乎/CSDN 的价格汇总文

**【估算】** YOLOX-nano/tiny 在 848 张图上微调 300 epoch：4090D 上大约 **0.5～2 小时**（单卡、bs=16～32、416～640）。**折算成本约 ¥1～4**，加上数据上传时间。**这是本次调研里性价比最高的一条结论：用 CPU 硬扛十几小时换来的东西，租卡 2 小时花 4 块钱就能拿到，而且质量更高。**

---

## 4. 依赖与许可风险清单

| 依赖 | 许可 | 商用/参赛注意点 |
| --- | --- | --- |
| **onnxruntime** | **MIT** | **最安全**。推理侧只依赖它最干净。注意：如果用了 `onnxruntime-gpu` 或 TensorRT EP，其打包的第三方组件另有条款，但常规 CPU 包是 MIT。 |
| **YOLOX**（Megvii-BaseDetection/YOLOX） | **Apache-2.0** | **推荐**。Apache-2.0 允许闭源商用，保留 NOTICE / LICENSE 即可。注意仓库里的 `yolox` PyPI 包装脚本引用了 AGPL 之外的组件（`onnx` 编译链），打包前建议只用推理部分。 |
| **PaddleDetection** | **Apache-2.0** | 可商用。但引入 PaddlePaddle 运行时会让部署体积与 CPU 推理耗时上一个大台阶，**不建议**。 |
| **MMDetection** | **Apache-2.0** | 可商用（仅限该仓库本身）。 |
| **MMYOLO** | **GPL-3.0**（GitHub API license 字段实测） | **风险项**。GPL-3.0 与闭源分发不兼容。**如果只是用它训练然后导出 ONNX，推理侧不带 MMYOLO 代码，通常可接受；但把 MMYOLO 源码并进产品是明确不行的。** |
| **RT-DETR** | 仓库许可随实现而异：PaddleDetection 版 Apache-2.0；另有独立实现需单独确认 | 用 PaddleDetection 版最省事。 |
| **Ultralytics YOLOv8 / YOLO11** | **AGPL-3.0** | **明确避开**。AGPL 的传染性对「软件对外提供服务」同样生效（网络服务条款），闭源商用需要买 Enterprise License。**同理：任何从 Ultralytics 权重微调出来的权重、以及 Roboflow 上导出的 YOLOv8 权重，都继承这一约束。** |
| **super-gradients (YOLO-NAS)** | Apache-2.0（仓库）；**但 YOLO-NAS 的权重另有 "YOLO-NAS license"**（非 Apache） | 权重许可需单独看，别默认跟着仓库走。 |
| 数据集（CPLID / InsPLAD 等） | 见第 1 节 | 数据集许可与代码许可**完全独立**，不能用「代码是 Apache」推断「数据能商用」。 |

**【判断】给工程的三条硬约束**

1. 产品里**不要出现 `ultralytics` 这个包名**，也不要在 `requirements.txt` 里留它。
2. 训练脚本（哪怕用了 MMYOLO）与推理产物（ONNX）**必须物理隔离**：交付物只包含 `.onnx` + `onnxruntime` 调用代码。
3. 每一个进入产品的权重，都要能回答「它是用什么代码、在什么数据上训出来的」，并把 LICENSE / NOTICE 留档。这对参赛材料同样是加分项。

---

## 5. 结论：三条路径

### 路径 A —— 最省事：只打通链路，不做真实训练

- **做什么**：下载 YOLOX-nano（Apache-2.0）的 COCO 权重 → 导出 ONNX → 把 `config/app.yaml` 的 `detect.detector` 从 `sidecar` 切到 ONNX 通道 → 用 `onnxruntime` + 现有 SAHI 切片跑通 → 在 CPLID / InsPLAD 上做一次域内评估，看切片参数是否需要调。
- **下载量**：权重约 **35 MB**（yolox_nano.pth）+ CPLID 约 **390 MB**（GitHub 镜像）或 InsPLAD 数 GB。合计 **0.5～5 GB**。
- **机时**：**不到 1 小时**（推理实测：nano@416 **43.2 ms/图 / 23.2 FPS**，nano@640 52.6 ms，tiny@416 48.4 ms，s@416 66.0 ms，单张 bs=1、12 线程）。
- **能得到什么级别的可信结论**：「ONNX 检测通道在生产配置下真的能跑」「SAHI 切片参数合理」「性能预算（单图几十毫秒）成立」。**不能**得到任何「能检出绝缘子缺陷」的结论——COCO 里没有 insulator 这个类。
- **代价**：零风险、零训练成本。**适合作为本周就能交付的第一步。**

### 路径 B —— 折中：租云 GPU 微调一个真正可用的权重

- **做什么**：在 AutoDL（或同类平台）租 4090D → 拉 YOLOX 官方代码 → 用 CPLID（2 类）或 InsPLAD（16 类）微调 YOLOX-tiny/S → 导出 ONNX → 拉回本机 → 走路径 A 的接入流程 → 在本机 CPU 上做推理验收与切片调参。
- **下载量**：数据集 **0.5～5 GB**（上传到云需同样流量）+ 权重 **十几 MB** + 环境约 3～5 GB。合计 **约 5～10 GB**。
- **机时**：云 GPU **0.5～3 小时**（约 **¥1～6**）；本机 CPU 推理验证 **1～2 小时**。
- **能得到什么级别的可信结论**：**可以在报告和演示里写「自训练权重，mAP@0.5 = X」**，X 由云端验证集给出。这是能进方案书、能进比赛材料的级别。风险点是数据集许可（第 1 节）。
- **代价**：云平台费用极低，主要成本是**数据合规**与**标注质量**。

### 路径 C —— 最彻底：自建数据闭环

- **做什么**：以路径 B 为起点，再补三件事——(1) 用真实巡检视频抽帧、自建标注（覆盖销钉缺失 / 均压环变形 / 锈蚀 / 自爆等 InsPLAD 里已有的细类）；(2) 把合成数据（CPLID 那 248 张合成缺陷图的思路）作为数据增广；(3) 建立「切片参数 × 输入尺寸 × 模型档位」的评测矩阵，对 SAHI 的 slice/overlap 做超参搜索。
- **下载量**：**10 GB 以上**（多数据集 + 自建数据）。
- **机时**：云 GPU **几十小时**（约 **¥100～300**）+ 本机评测数十小时。
- **能得到什么级别的可信结论**：**可对外交付的检测能力**，能写出「在自建测试集上 mAP@0.5 = X / 召回 = Y」并且能解释失效模式。这也是唯一能支撑「真实缺陷检出率」这类对外承诺的路径。
- **代价**：标注人力是主要成本，不是算力。

**【判断】推荐顺序：本周先做 A（1 小时，把通道坐实）→ 下周做 B（几块钱，拿到可信权重）→ 按产品节奏决定是否做 C。** 不要在 CPU 上硬训，那是拿几天机时换一个连评估都不可信的模型。

---

## 附录 A：本次实测的环境与复现方式

- 机器：i5-12500H（12 物理核 / 16 线程），Windows，物理内存 15.8 GB，调研期间可用约 4.7 GB
- Python：3.12（项目内临时虚拟环境），torch 2.x CPU 版 + torchvision CPU 版，线程数固定 12
- 基准内容分布：
  - `yolox_impl.py`：逐行复刻 YOLOX 的 `network_blocks.py` / `darknet.py` / `yolo_pafpn.py` / `yolo_head.py` 训练前向，用于参数量与耗时对齐
  - `_bench_yolox_real.py`：训练步 / 推理耗时基准（参数、FLOPs、ms/img）
  - `_bench_head_cost.py`：SimOTA（`yolox_utils.py` 全流程复刻）与 Mosaic（`mosaicdetection.py` 主路径复刻）的 CPU 开销
  - `_bench_mem.py` / `_probe_mem.py`：峰值工作集测量（`GetProcessMemoryInfo` 的 `PeakWorkingSetSize`）
- **数据边界说明（重要）**：
  - 第 3.3 / 3.4 / 3.5 节的耗时数据来自**结构等价复刻**，不是官方 YOLOX 训练器的端到端计时。差异可能来自：官方 dataloader 的多进程开销、Mosaic/HSV 等增广的完整实现、验证阶段的额外前向、以及官方 trainer 的日志/EMA 开销。**这些都会让真实训练比上表更慢，不会更快。**
  - SimOTA 的耗时随 GT 数呈平方级上升，本机在「640 输入 + 30 GT」这一格**因内存不足未能测完**（进程崩溃，`DefaultCPUAllocator: not enough memory`，一次尝试分配约 7.5 GB）。因此那格的数字标为「未完成」而不是给一个估计值。
  - 官方 YOLOX **未能 pip 安装**（PyPI 包需要 cmake 编译 onnx 扩展，本机无 cmake；也无法从 github.com clone）。所有源码是通过 GitHub REST API（`contents` 接口，返回 base64）逐文件取回的。

## 附录 B：调研中受阻、需要人工确认的三件事

1. **CPLID 与 InsPLAD 的许可条款**：IEEE DataPort 与 Mendeley Data 的条目页面在未登录状态下没有显示明确的 license 字段。**需要人工登录后核对，或直接联系作者**。这两个数据集的商用可行性目前**悬空**。
2. **GitHub Releases 在本机不可直连**：`github.com` 域内网络时通时断（API 可通、raw/网页不可通）。下载权重和数据时需要有代理或镜像方案，否则方案书里写的下载步骤在客户现场可能跑不通。
3. **YOLOX 官方权重的 ONNX 导出**：本机没有 cmake，无法安装官方 `yolox` 包，因此**导出这一步没有实测**。需要在有 cmake 或 Linux 环境上验证一次（或直接用官方 `demo/ONNXRuntime/` 的导出脚本）。

## 附录 C：一条容易被忽略的接口约束

`loweye/detect/onnx_detector.py` 目前的推理链路假定输出可以被解码成 `[x1, y1, x2, y2, score, class]` 的候选框。换成 YOLOX 时，ONNX 图默认输出的是 **raw head 输出**（`(1, N, 5+nc)`，且带网格偏移），需要：

1. 在 `onnx_detector.py` 里实现 YOLOX 的 decode（参考 `demo/ONNXRuntime/onnx_inference.py` 的 `demo_postprocess`），或
2. 在导出时把 decode 一起固化进 ONNX 图（官方 `export_onnx.py` 支持，但会固化输入尺寸与类别数）。

**【判断】选 (1)**：把 decode 留在 Python 侧，换尺寸/换类别数时不必重新导出。这个改动量很小（几十行），但必须先做，否则接上权重也跑不出框。
