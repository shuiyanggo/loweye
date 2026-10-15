# 第五章附二　检测环节：真实权重通道与逐类漏检率

> 前两章附讲的是检索。这一章讲检测——它此前是全篇最空的一环：检测后端读的是同名 `.detections.json`（sidecar 合成标注），只能证明「链路连得通」，不能证明「模型看得见」。本章把真实权重通道接上，并给出第一批可复现的检测数字，包括一个必须直面的负面结果。

## 附二 1 通道：从 sidecar 换成真实 ONNX 权重

| 项 | 内容 |
| --- | --- |
| 推理后端 | `onnxruntime`（CPU）+ 项目既有 SAHI 切片（640 瓦片、overlap 0.2），跨片 NMS |
| 权重 | `models/detector/rtmdet-tiny-coco.onnx`（21.3 MB），模型本体是 OpenMMLab **RTMDet-tiny**，**Apache-2.0**；第三方 ONNX 导出 |
| 为什么不选 YOLOv8/v5 | 那些公开「绝缘子权重」几乎都是 Ultralytics 系训练产物，继承 **AGPL-3.0**，与作品的许可要求冲突（见 `_research_detection.md` 第 2 节） |
| 配置 | `config/app.onnx.yaml`（独立文件，不动默认的 sidecar 配置） |
| 导出格式适配 | MMDeploy 系是**双输出** `dets(1,N,5)=[x1,y1,x2,y2,score]` + `labels(1,N)`，与 YOLO 单输出 `(1,4+nc,N)` 不同；`loweye/detect/onnx_detector.py` 现按输出名自动判定布局并分别解码，遇到不认识的形状直接报错而不是猜 |

**阳性对照（这张是必须有的）**：对一张真实照片（960×686）跑出 `cat` 两个框（score 0.651 / 0.630，534 ms，4 个切片）；对合成演示图跑出 0 个目标——因为 COCO 80 类里根本没有「输电杆塔」。前者证明通道真的在检测，后者说明「0 检出」在这个模型上不代表链路坏了。【实测】

## 附二 2 域内实测：COCO 权重在 CPLID 上的逐类漏检率

评估数据用 **CPLID**（中国电力线绝缘子数据集，848 张：600 正常 + 248 缺陷，Pascal VOC 标注，1152×864 真实航拍图）。本次取子集各 50 张，逐类结果：

| 数据集 | GT 类别 | GT 框数 | 命中 | **逐类漏检率** | **类无关召回** | 空图率 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 缺陷图 50 张 | `defect` | 50 | 0 | **100%** | **0 / 50 = 0%** | 74% | 428 ms（4.0 切片） |
| 正常图 50 张 | `insulator` | 101 | 0 | **100%** | **0 / 101 = 0%** | 76% | 494 ms（4.0 切片） |

模型在这 100 张图上总共只输出了 42 个框，类别是 `person` / `airplane` / `clock` / `bed` / `bench` / `umbrella` / `truck` / `car` / `bicycle` / `toilet` / `traffic light` —— **一个绝缘子都没框到**。

两个口径的区别必须讲清楚：**逐类漏检率 100% 是必然的**（COCO 里没有绝缘子类别，标签对不上就不算命中）；所以真正有信息量的是**类无关召回 0%**——它忽略类别标签，只问「有没有任何一个框压到绝缘子上」，答案是**一个都没有**。【实测】

## 附二 3 自训练：在 CPLID 上微调一个缺陷权重

COCO 权重零召回证明「检测环必须自训练」。下面是把这件事做完的记录——包括为什么换模型、数据怎么切、训了多少、留出集上什么结果。

### 3.1 为什么最后用 FCOS-ResNet50-FPN

| 候选 | 结论 | 原因 |
| --- | --- | --- |
| MMDetection / RTMDet-tiny | **放弃** | 预编译 `mmcv` 只到 cu121/torch2.2，租用机是 torch 2.8+cu128；机器无 `nvcc`，不能源码编译。为它降级重装 torch 会把整台机器的环境搞脏 |
| Ultralytics YOLO 系 | **不用** | 与附二 1 同一条许可理由（AGPL-3.0） |
| **torchvision `fcos_resnet50_fpn`** | **采用** | 与 RTMDet 同属 anchor-free 单阶段检测器；BSD-3-Clause；`torch.onnx.export` 直接可导；COCO 预训练权重是官方发布的（文件名 `fcos_resnet50_fpn_coco-99b0c9b7.pth`） |

训练脚本 `tools/gpu/train_cplid_fcos.py`：COCO 预训练 → 重建 `head.classification_head.cls_logits` 为 `num_anchors × 3`（背景 / `insulator` / `defect`，初始化 std 0.01、bias −log((1−0.01)/0.01)）→ `model.transform.min_size/max_size` 与输入尺寸对齐（**不这么做导出的 ONNX 会把尺寸当常量，喂别的尺寸静默算错**）。自写 letterbox（灰边 114）把 1152×864 映射到 1024，训练时 50% 水平翻转；SGD momentum 0.9、wd 5e-4、cosine 退火到 lr×0.02、梯度裁剪 10。

### 3.2 数据与算力

| 项 | 值 |
| --- | --- |
| 数据 | CPLID 848 张：600 正常（标注 `labels/*.xml`，类 `insulator`）+ 248 缺陷（标注 `labels/defect/*.xml`，类 `defect`，每图 1 框） |
| 划分 | `out/cplid_split.json`（seed 20261013，val_ratio 0.2）→ 训练 737 张、留出 111 张（63 正常 + 48 缺陷） |
| 留出集与其它档一致 | 该划分与本机词法/向量两档、以及两段式分类器用的是**同一批留出图**，训练时未见过 |
| 算力 | 租用 AutoDL 单卡 **RTX 4090D 49 GB**（¥1.88/小时），torch 2.8.0+cu128 |
| 训练 | 30 轮、batch 6、size 1024、workers 12，约 **36.4 秒/轮 → 全程约 18 分钟**（GPU 利用率 92~98%） |
| 产物 | `cplid_fcos_best.pth`（122.8 MB）→ 导出 `cplid_fcos_1024.onnx`（122.9 MB，opset 16，三输出 `boxes/labels/scores`） |

训练过程中每轮同时记录 `miss_insulator` / `miss_defect` / `recall_all`，第 1 轮起 `miss_defect` 就是 0.0%，`miss_insulator` 稳定在 1.8%~2.7%。

### 3.3 留出集结果（`tools/eval_detection.py`，IoU ≥ 0.5）

| 留出集 | 类别 | GT 框 | 命中 | **逐类漏检率** | **类无关召回** | 空图率 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 缺陷图 48 张 | `defect` | 48 | 48 | **0%** | **100%** | 0% | 916 ms |
| 正常图 63 张 | `insulator` | 113 | 110 | **3%** | **97%** | 0% | 1081 ms |

同一批数据上，COCO 权重的类无关召回是**0%**，微调后是 **100% / 97%**。【实测】

**必须一起说的误报**：缺陷留出集里除 48 个命中框外，另有 16 个 `defect` 误报与 122 个 `insulator` 误报；正常留出集里 113 个 GT 之外有 97 个 `insulator` 重复框、3 个 `defect` 误报。也就是说这套权重**召回很强、精度一般**，工程上必须靠「正常类不出工单」与人工确认两道闸门兜住（见 3.4）。只报召回不报误报，是耍流氓。【实测】

## 附二 4 接回流水线：三处「权重能跑但系统不对」的地方

把 ONNX 接回 `config/app.cplid_gpu.yaml` 后，逐项验证时抓到三个**不会报错、只会算错**的问题：

1. **类别数不匹配**：配置里的 `num_classes` 还是 COCO 的 80，而模型是 3 类。ONNX 只按形状解码，多出来的类不会抛异常，只会让标签含义整体错位。→ 在配置里显式写 `num_classes: 3` 与 `class_names: [insulator, defect]`。
2. **阈值读了模块级配置**：`loweye/detect/onnx_detector.py` 里 `conf_threshold` 走的是模块顶部加载的默认配置，而不是流水线传进来的那份配置——表现为「配置里改了阈值，服务上不生效」。→ 改为从传入的 `cfg` 读，与其它后端一致。
3. **背景占了一个通道**：torchvision 系导出的 `labels` 从 **1** 开始（0 是背景），直接拿去查类别名会整体错位一位。→ 在 `loweye/detect/onnx_detector.py` 新增 `tv_dets` 布局分支（按输出名 `boxes/labels/scores` 自动判定），标签减一后再映射，超范围标签回落成 `class_N` 而不是崩。配套单测 `tests/test_onnx_layout.py`（6 项）覆盖：标签映射、letterbox 还原、低分过滤、超范围回落、形状不符抛错、带 batch 维。

还有一处不是检测的问题、但会让缺陷「检出了却升不了级」：分级规则表 `config/routes.yaml` 里**没有 `defect` / `insulator` 两个类别**，于是真实权重检出的一切都会走默认等级并全部转人工——现场表现就是「检出 0.82 也判 P3」。已补两条规则：

| 类别 | 规则 | 语义 |
| --- | --- | --- |
| `defect` | **P1 绝缘子破损**、输电运检班、SLA 24 h、强制人工确认 | 缺陷即刻成单，链路最短 |
| `insulator` | **P4 正常**、`normal: true` | 出结论、**不出工单**、不检索条款 |

`normal: true` 这个开关是必需的：真实权重必然会把健康的绝缘子也检出来（一张图几串到十几串），若每串都成单，工单队列会被正常设备淹掉，真正的缺陷反而看不见。同时正常类不检索条款——否则必然召回「缺陷处置」类条款，套到正常设备上就是**伪依据**。

**端到端实测**（`config/app.cplid_gpu.yaml`，服务端口 8851，上传真实航拍图）：

| 输入 | 检出 | 诊断 | 工单 |
| --- | --- | --- | --- |
| 缺陷图 `Defective_Insulators/000.jpg` | `defect` 0.825 + `insulator` 0.332 / 0.257 | 绝缘子破损 **P1**（5 条依据）、绝缘子（未见缺陷）P4 ×2 | **1 张**：P1 绝缘子破损（0.82）→ 输电运检班、SLA 24 h、`route=human_review` |
| 正常图 `Normal_Insulators/0049 / 0051 / 0056.jpg` | `insulator` 2 / 4 / 2 个（0.28~0.72） | 全部「绝缘子（未见缺陷）」P4 | **0 张** |

【历史实测记录】该 CPLID 接入轮次单测同步扩到 **130 项全过**（新增类别覆盖、正常类不成单两条守卫）；本次文档同步未重跑全量测试，不代表当前测试数量或通过状态。

## 附二 5 这一环现在能主张什么

1. **能主张**：一条真实可复现的「自训练权重 → ONNX → 领域类别 → 分级规则 → 工单」通路，缺陷留出集逐类漏检率 **0%**、正常留出集 **3%**，端到端行为符合设计（缺陷成单、正常不成单）。【实测】
2. **不能主张**：绝对精度。数据只有 CPLID 848 张、且其许可未明示；误报率不低（见 3.3）；换到别的杆塔、别的拍摄角度、别的缺陷类型（自爆之外的破损/异物/污闪）都不在验证范围内。这些是**当前版本的能力边界**，不是可以含糊过去的地方。【判断】
3. **下一步最该做的**：用 InsPLAD（10607 张、16 类缺陷）扩充训练集，并把误报压下来——现阶段的 `insulator` 重复框说明后处理（跨片/同类 NMS 或置信度阈值）还有明确空间。【判断】


## 附二 6 复现

```powershell
# 1) 拉一张 Apache-2.0 的 COCO 权重（第三方 ONNX 导出，21 MB）
python -m tools.probe_detector --config config/app.onnx.yaml `
  --image models/detector/samples/sample_cat2.jpg      # 阳性对照：应检出 cat

# 2) 拉 CPLID 子集（HF 镜像；仓库无 LICENSE，仅本地评测用，不随作品分发）
python -m tools.fetch_cplid --limit 50

# 3) 逐类漏检率（COCO 权重的负面结果）
python -m tools.eval_detection --config config/app.onnx.yaml `
  --images data/external/cplid/Defective_Insulators/images `
  --labels data/external/cplid/Defective_Insulators/labels/defect `
  --out out/eval_detection_coco_on_cplid_defect.md

# 4) 自训练权重（留出集，与其它档同一划分）
python -m tools.eval_detection --config config/app.cplid_gpu.yaml `
  --images data/external/cplid/Defective_Insulators/images `
  --labels data/external/cplid/Defective_Insulators/labels/defect `
  --only-from out/cplid_split.json --split-key val `
  --out out/eval_detection_fcos_defect.md --json out/eval_detection_fcos_defect.json

python -m tools.eval_detection --config config/app.cplid_gpu.yaml `
  --images data/external/cplid/Normal_Insulators/images `
  --labels data/external/cplid/Normal_Insulators/labels `
  --only-from out/cplid_split.json --split-key val `
  --out out/eval_detection_fcos_normal.md --json out/eval_detection_fcos_normal.json

# 5) 起真权重的服务（默认档仍是 sidecar，端口不冲突）
python -m loweye.cli.serve --config config/app.cplid_gpu.yaml --host 127.0.0.1 --port 8851
```

**数据许可声明**：CPLID 由 Tao et al. 随论文公开，仓库与数据平台条目均未标明 LICENSE。本作品只在本地做算法验证并署名来源，**不随作品再分发数据集**；商用前须向数据方确认。【事实】

## 附二 7 烟火 / 农作物害虫扩展进度与边界

当前默认配置仍是 sidecar 演示，已有 CPLID GPU 自训练真实权重及其留出集实测结果是另一条独立通道；本节扩展不继承 CPLID 的检测指标。这里“烟火”指 **烟雾 / 火焰**，不是烟花。**两个扩展模型均已完成 24 轮正式训练：农作物害虫（crop_pest / IP102）与烟火（fire_smoke / D-Fire）的 best / last 与 ONNX 均已按 SHA-256 校验回收，并完成独立 ONNX 复评。逐轮曲线、权重选型与测试集指标以 `docs/proposal/06d_trained_detectors.md` 为准，本节不重复。**

| 数据源 / 环节 | 已验证状态（含历史失败记录） | 证据 |
| --- | --- | --- |
| IP102 检测子集 | 下载哈希核验通过：**807750864 字节**；SHA-256：`38c8a26152baeea9283d593a3ab955e8fe3258105d08a0e30372765111f57fb0` | `out/inspection_gpu/ip102_transport.json`、`out/inspection_gpu/ip102_source_license.json` |
| IP102 首次准备（历史） | **失败 180 项**：174 项源数据使用 0 基类号（初版转换按 1..102 校验而拒绝）、5 项缺标注、1 项拼接 XML；该轮未通过，不把哈希通过当作准备完成 | `out/inspection_gpu/ip102_initial_preparation_errors.json` |
| IP102 第二次准备（历史） | 仍失败：**7 项 XML / TXT 双标注冲突**；该轮已隔离首次发现的 **6 项源问题**（5 缺标注 + 1 拼接 XML） | `out/inspection_gpu/ip102_second_preparation_errors.json` |
| IP102 第三次准备与原源审计（历史） | 第三次因缺少前景类号 **60 / 61 / 64 / 76 / 81** 未通过；隔离累计 13 项。隔离 / 去重前的全部源标注审计：XML **18976 份、有效 18975 份**，TXT **7 份**；合并统计原源本就缺少 0 基类号 **59 / 60 / 63 / 75 / 80**，不是隔离或去重造成丢类 | `out/inspection_gpu/ip102_third_preparation_errors.json`、`out/inspection_gpu/ip102_source_class_audit.json` |
| IP102 准备通过 | 显式允许源缺类并隔离坏标注后通过：**102 类表保留、train 12894 / val 2276 / test 3798（官方 test 划分保留）**，`voc_class_id_base=0` 映射到内部 1..102，隔离 13 项，无跨划分泄漏 | 运行时清单 `out/inspection_gpu/crop_pest_manifest.json` |
| D-Fire 下载、解包与固定划分 | 下载 **3049605157 字节**（MD5 与源站 ETag 一致、SHA-256 `65d81b60…`）；解包 **43055 项 / 3118334483 字节**；严格模式首跑报 **340 项**越界 / 非有限框，显式隔离后固定划分 **train 13898 / val 3045 / test 4244**，无跨划分泄漏 | `out/inspection_gpu/dfire_transport.json`、`out/inspection_gpu/dfire_source_license.json`、`out/inspection_gpu/dfire_strict_preparation_errors.json` |
| 扩展训练与回收 | 两模型冒烟门控均已通过；crop_pest 与 fire_smoke 均已完成 24 轮训练，两者的 ONNX 与 best / last 均已按 SHA-256 校验回收并完成独立 ONNX 复评 | `out/inspection_gpu/crop_pest_recovery.json`、`out/inspection_gpu/crop_pest_onnx_eval.json`、`models/detector/crop_pest_fcos_800.onnx`、`out/inspection_gpu/fire_smoke_recovery.json`、`out/inspection_gpu/fire_smoke_onnx_eval.json`、`models/detector/fire_smoke_fcos_640.onnx` |

实施调整为显式允许源数据缺类并**保留 102 类别表**，隔离 13 项坏标注后准备通过；但**保留类别表不等于缺失类可训练**：上述缺失 **5 类（0 基类号 59 / 60 / 63 / 75 / 80）没有源样本，无法训练也无法评估**，因此不能宣称模型具备全部 102 类识别能力；测试集上的逐类指标对它们只能标为不可评估。原源审计计数包含双标注，不是去重后的独立目标数。

**实现许可、数据许可与权重许可分别核对**：`torchvision` / FCOS 的 **BSD-3-Clause 仅是实现许可**，不代表训练数据或派生权重可分发。

- **IP102** 官方来源为 [xpwu95/IP102](https://github.com/xpwu95/IP102)：学术用途免费，其他用途须联系作者；镜像宣称的 PDDL 不覆盖官方条款，**竞赛 / 商业用途及数据、权重分发许可尚未确立**。本项目不再分发原始数据。它是农作物害虫数据，**不是植物病害数据，也未完成航拍验证**。来源工具为 `tools/fetch_ip102.py`，许可记录为 `out/inspection_gpu/ip102_source_license.json`。
- **D-Fire** 官方来源为 [DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset)：集合标示 **CC0-1.0**，但上游不主张拥有全部原始图像权利，不能将集合许可解读为全部原图权利已清理；本项目不再分发原图。其类别为烟雾 / 火焰，**不包含烟花标注，未完成航拍验证**。来源与披露见 `tools/fetch_dfire.py`。

本次同步文档与现有证据：全量单元测试实测 **264 项全绿**（2026-10-15，21.237 秒，退出码 0）；118 / 130 项测试数字均为既往轮次的历史统计，不作为本次通过状态。
