# 训练交付：fire_smoke 与 crop_pest 检测器

本文件由 `tools/write_model_report.py` 从回收制品自动生成。所有数字直接读自 JSON/JSONL，不手工誊写、不四舍五入到更「好看」的值；缺失的输入会显式标注为缺失并列在文末的「证据缺口」中。

生成时间：2026-10-15T09:03:47

## 0. 结论速览

| 任务 | 训练器自测 mAP | 独立 ONNX 复评 mAP | 绝对差 | 测试图像数 | 最优轮 |
| --- | --- | --- | --- | --- | --- |
| fire_smoke | 0.371005 | 0.371026 | 0.000020 | 4244 | 20 |
| crop_pest | 0.232539 | 0.232567 | 0.000029 | 3798 | 17 |

两套实现相互独立：训练器走 torchmetrics(backend=pycocotools)，复评走 onnxruntime + 直接调用 pycocotools COCOeval；两者共用同一测试划分与同一 ONNX 权重。绝对差超过 0.005 会被标记为告警（本次：无）。

**本文不含任何跨数据集、跨场景或跨机型的泛化结论**：数字只描述下列测试划分。

### fire_smoke — D-Fire 烟火/明火检测

用途定位：低空巡检中的烟火与明火告警。推理尺寸 640×640，输出为 letterbox 坐标系下的 xyxy 框 + 1 基类别 id + 分数，需按 metadata 中的预处理说明还原到原图坐标。

#### 数据来源与划分

| 项目 | 值 |
| --- | --- |
| 数据源 | D-Fire（dfire） |
| 官方来源 | https://github.com/gaia-solutions-on-demand/DFireDataset |
| 归档字节 | 3049605157 (2908.3 MiB) |
| 归档 sha256 | 65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83 |
| 类别数 | 2 |
| 标注格式 | {"YOLO_0based_normalized": 21187} |
| 类别 id 基数（源） | 1 |
| 划分策略 | official split preserved |
| seed | 42 |
| 按字节去重移除 | 0 |
| 跨划分同源分组（应空） | [] |

各划分规模（来自 manifest.statistics，不是原始源计数）：

| 划分 | 图像数 | 无目标图像 | 同源分组数 | 目标框合计 |
| --- | --- | --- | --- | --- |
| train | 13898 | 6458 | 13898 | 16611 |
| val | 3045 | 1375 | 3045 | 3709 |
| test | 4244 | 2005 | 4244 | 4993 |

隔离的无效源样本（未移动或修改任何源文件）：

| 项目 | 值 |
| --- | --- |
| 启用 | true |
| 数量 | 340 |
| 按原划分 | {"test": 62, "train": 224, "val": 54} |
| 按原因 | {"ValueError: line 3: nonfinite/out-of-range coordinates or empty box": 4, "ValueError: line 1: box extends beyond image": 121, "ValueError: line 5: box extends beyond image": 25, "ValueError: line 3: box extends beyond image": 40, "ValueError: line 2: box extends beyond image": 71, "ValueError: line 2: nonfinite/out-of-range coordinates or empty box": 11, "ValueError: line 5: nonfinite/out-of-range coordinates or empty box": 5, "ValueError: line 4: box extends beyond image": 36, "ValueError: line 6: box extends beyond image": 12, "ValueError: line 8: box extends beyond image": 5, "ValueError: line 1: nonfinite/out-of-range coordinates or empty box": 4, "ValueError: line 7: box extends beyond image": 4, "ValueError: line 7: nonfinite/out-of-range coordinates or empty box": 1, "ValueError: line 8: nonfinite/out-of-range coordinates or empty box": 1} |

#### 训练配置（读自 run.json 的 signature）

| 项目 | 值 |
| --- | --- |
| 优化轮数 | 24 |
| 批大小 | 4 |
| 学习率 | 0.0002 |
| 数据加载 worker | 8 |
| 随机种子 | 20261013 |
| 训练样本上限 | 0 |
| 验证样本上限 | 0 |
| manifest sha256 | efa99a1d465ad9bf7fdbac54eef241afd1d14db676625242c55e9f1cfbe241c7 |
| 冒烟运行 | false |
| 架构 | torchvision FCOS ResNet50 FPN |
| 实现许可 | BSD-3-Clause |
| torch / onnxruntime | {"torch": "2.8.0+cu128", "onnxruntime": "1.31.0"} |

#### 模型选择

| 项目 | 值 |
| --- | --- |
| 选择划分 | val |
| 选择准则 | COCO map |
| 入选轮次 | 20 |
| 入选轮 mAP(0.50:0.95) | 0.375623 |
| 入选轮 mAP@.50 | 0.709094 |

测试集不参与任何模型选择：训练器只在保存最优权重之后才触碰一次测试划分，并写入不可变收据（重复评估会被 signature + checkpoint sha256 拒绝）。

#### 独立测试（训练器实现：torchmetrics + pycocotools）

| 指标 | 值 |
| --- | --- |
| 测试状态 | evaluated once |
| mAP(0.50:0.95) | 0.371005 |
| mAP@.50 | 0.699386 |
| 全类召回 | 0.769277 |
| 误检占报出比例 | 0.516003 |
| 每图误报数 | 0.964892 |
| 收据评估次数 | 1 |
| 收据 checkpoint sha256 | 721646ee07734922 |

#### 独立复评（本项目 ONNX 评测器：onnxruntime + 直接调用 pycocotools COCOeval）

| 指标 | 值 |
| --- | --- |
| 测试图像数 | 4244 |
| 类别数 | 2 |
| mAP(0.50:0.95) | 0.371026 |
| mAP@.50 | 0.699423 |
| 真值框总数 | 4993 |
| 0.001 阈值以下检出数 | 1273200 |
| 可评估类别数 | 2 |
| 不可评估类别数 | 0 |
| 全类召回 | 0.769477 |
| 每图误报数 | 0.965834 |
| onnxruntime | 1.31.0 |
| 执行提供者 | ["CPUExecutionProvider"] |
| 预处理一致性（与训练器数据集类逐元素比较） | {"images_compared": 8, "max_abs_difference": 0.0, "reference": "train_inspection_fcos.InspectionDataset", "manifest": "/root/loweye/data/fire_smoke_manifest.json", "agrees_within_1e-6": true} |
| 耗时 | 529.5 秒 / 0.125 秒每图 |
| ONNX 字节 / sha256 | 128908889 / c4149315fae521af |

独立复评可评估类别中 mAP 最高与最低各若干：

| 类别 | mAP | 真值框数 | 方向 |
| --- | --- | --- | --- |
| smoke | 0.441227 | 2253 | 最高 |
| fire | 0.300824 | 2740 | 最高 |

#### 逐轮曲线（训练器每轮 fsync 写入的 JSONL）

| 项目 | 值 |
| --- | --- |
| 已记录轮数 | 24 |
| 首轮 mAP | 0.140420 |
| 末轮 mAP | 0.369349 |
| 末轮 mAP@.50 | 0.702837 |
| 末轮 loss | 0.7893 |
| 最优轮（按 val mAP） | 20 |
| 最优轮 mAP | 0.375623 |
| 平均每轮耗时 | 419.2 秒 |
| 累计训练耗时 | 10059.9 秒 |

#### 回收产物（本地重建自远端，逐文件与远端 sha256sum 比对）

| 文件 | 本地字节 | sha256 前 16 位 |
| --- | --- | --- |
| fire_smoke_fcos_640.onnx | 128908889 (122.9 MiB) | c4149315fae521af |
| fire_smoke_fcos_metadata.json | 6041 (0.0 MiB) | 1ae051628da99f08 |
| fire_smoke_fcos_best.pth | 383676643 (365.9 MiB) | 721646ee07734922 |
| fire_smoke_fcos_last.pth | 383676643 (365.9 MiB) | ba7682e5a53a790e |
| fire_smoke_test.json | 1835 (0.0 MiB) | f496de357d5c7e83 |
| fire_smoke_run.json | 958 (0.0 MiB) | ea0160878f5ab5f5 |
| fire_smoke_manifest.json | 13298926 (12.7 MiB) | 7e64f84ff225f78f |
| train_fire_smoke.log | 70149 (0.1 MiB) | c1b7ccf0b1979fbc |

### crop_pest — IP102 农业害虫检测

用途定位：农田虫情普查与靶向施药辅助。推理尺寸 800×800，输出为 letterbox 坐标系下的 xyxy 框 + 1 基类别 id + 分数，需按 metadata 中的预处理说明还原到原图坐标。

#### 数据来源与划分

| 项目 | 值 |
| --- | --- |
| 数据源 | IP102 detection subset（ip102） |
| 官方来源 | https://github.com/xpwu95/IP102 |
| 归档字节 | 807750864 (770.3 MiB) |
| 归档 sha256 | 38c8a26152baeea9283d593a3ab955e8fe3258105d08a0e30372765111f57fb0 |
| 类别数 | 102 |
| 标注格式 | {"VOC_0based_inclusive": 18968} |
| 类别 id 基数（源） | 0 |
| 划分策略 | official test preserved; seed42 grouped 15% validation from official training pool |
| seed | 42 |
| 按字节去重移除 | 0 |
| 跨划分同源分组（应空） | [] |

各划分规模（来自 manifest.statistics，不是原始源计数）：

| 划分 | 图像数 | 无目标图像 | 同源分组数 | 目标框合计 |
| --- | --- | --- | --- | --- |
| train | 12894 | 0 | 12894 | 15152 |
| val | 2276 | 0 | 2276 | 2680 |
| test | 3798 | 0 | 3798 | 4444 |

隔离的无效源样本（未移动或修改任何源文件）：

| 项目 | 值 |
| --- | --- |
| 启用 | true |
| 数量 | 13 |
| 按原划分 | {"unsplit": 5, "train": 8} |
| 按原因 | {"ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP009000231.jpg: found 0": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP014000238.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP014000841.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP015000343.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP015001157.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP015001158.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP015001159.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP015001371.jpg: found 2": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP022000163.jpg: found 0": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP034000029.jpg: found 0": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP048000152.jpg: found 0": 1, "ValueError: Missing or ambiguous annotation for /root/loweye/data/external/ip102/JPEGImages/IP053000086.jpg: found 0": 1, "ParseError: junk after document element: line 27, column 0": 1} |

#### 训练配置（读自 run.json 的 signature）

| 项目 | 值 |
| --- | --- |
| 优化轮数 | 24 |
| 批大小 | 4 |
| 学习率 | 0.0002 |
| 数据加载 worker | 8 |
| 随机种子 | 20261013 |
| 训练样本上限 | 0 |
| 验证样本上限 | 0 |
| manifest sha256 | c6dc9be48804be633c5f1332054d47611a11c549940f2a144917ba164d495655 |
| 冒烟运行 | false |
| 架构 | torchvision FCOS ResNet50 FPN |
| 实现许可 | BSD-3-Clause |
| torch / onnxruntime | {"torch": "2.8.0+cu128", "onnxruntime": "1.31.0"} |

#### 模型选择

| 项目 | 值 |
| --- | --- |
| 选择划分 | val |
| 选择准则 | COCO map |
| 入选轮次 | 17 |
| 入选轮 mAP(0.50:0.95) | 0.252823 |
| 入选轮 mAP@.50 | 0.422896 |

测试集不参与任何模型选择：训练器只在保存最优权重之后才触碰一次测试划分，并写入不可变收据（重复评估会被 signature + checkpoint sha256 拒绝）。

#### 独立测试（训练器实现：torchmetrics + pycocotools）

| 指标 | 值 |
| --- | --- |
| 测试状态 | evaluated once |
| mAP(0.50:0.95) | 0.232539 |
| mAP@.50 | 0.392214 |
| 全类召回 | 0.795005 |
| 误检占报出比例 | 0.760312 |
| 每图误报数 | 2.950764 |
| 收据评估次数 | 1 |
| 收据 checkpoint sha256 | 8a599154af1a1a51 |

#### 独立复评（本项目 ONNX 评测器：onnxruntime + 直接调用 pycocotools COCOeval）

| 指标 | 值 |
| --- | --- |
| 测试图像数 | 3798 |
| 类别数 | 102 |
| mAP(0.50:0.95) | 0.232567 |
| mAP@.50 | 0.392283 |
| 真值框总数 | 4444 |
| 0.001 阈值以下检出数 | 1139320 |
| 可评估类别数 | 96 |
| 不可评估类别数 | 6 |
| 全类召回 | 0.794779 |
| 每图误报数 | 2.950500 |
| onnxruntime | 1.31.0 |
| 执行提供者 | ["CPUExecutionProvider"] |
| 预处理一致性（与训练器数据集类逐元素比较） | {"images_compared": 8, "max_abs_difference": 0.0, "reference": "train_inspection_fcos.InspectionDataset", "manifest": "/root/loweye/data/crop_pest_manifest.json", "agrees_within_1e-6": true} |
| 耗时 | 972.6 秒 / 0.256 秒每图 |
| ONNX 字节 / sha256 | 129830891 / 425cfd8a7920a420 |

独立复评可评估类别中 mAP 最高与最低各若干：

| 类别 | mAP | 真值框数 | 方向 |
| --- | --- | --- | --- |
| Ampelophaga | 0.701344 | 56 | 最高 |
| Cicadellidae | 0.668567 | 593 | 最高 |
| Lycorma delicatula | 0.654308 | 68 | 最高 |
| grub | 0.548838 | 118 | 最高 |
| mole cricket | 0.518999 | 176 | 最高 |
| Chlumetia transversa | 0.029592 | 12 | 最低 |
| Limacodidae | 0.025279 | 47 | 最低 |
| wheat blossom midge | 0.022246 | 6 | 最低 |
| Aphis citricola Vander Goot | 0.002041 | 1 | 最低 |
| therioaphis maculata Buckton | 0.000549 | 3 | 最低 |

#### 逐轮曲线（训练器每轮 fsync 写入的 JSONL）

| 项目 | 值 |
| --- | --- |
| 已记录轮数 | 24 |
| 首轮 mAP | 0.018458 |
| 末轮 mAP | 0.246357 |
| 末轮 mAP@.50 | 0.417115 |
| 末轮 loss | 0.7791 |
| 最优轮（按 val mAP） | 17 |
| 最优轮 mAP | 0.252823 |
| 平均每轮耗时 | 494.4 秒 |
| 累计训练耗时 | 11866.2 秒 |

#### 回收产物（本地重建自远端，逐文件与远端 sha256sum 比对）

| 文件 | 本地字节 | sha256 前 16 位 |
| --- | --- | --- |
| crop_pest_fcos_800.onnx | 129830891 (123.8 MiB) | 425cfd8a7920a420 |
| crop_pest_fcos_metadata.json | 86521 (0.1 MiB) | 1cb5c15ba2afe9c4 |
| crop_pest_fcos_best.pth | 386456891 (368.6 MiB) | 8a599154af1a1a51 |
| crop_pest_fcos_last.pth | 386456891 (368.6 MiB) | ab16ca926c93c959 |
| crop_pest_test.json | 42321 (0.0 MiB) | 273d7e536bfe73b4 |
| crop_pest_run.json | 13860 (0.0 MiB) | b972e4d75383ff15 |
| crop_pest_manifest.json | 11328662 (10.8 MiB) | 420f818f484715fc |
| train_crop_pest.log | 796974 (0.8 MiB) | b63b89b56a5a5aba |

## 附录：能力边界与许可（逐条附来源）

**fire_smoke**

- 类别覆盖策略（来源 `fire_smoke_manifest.json` 的 `class_coverage.policy`）：`strict_all_classes_observed`。
- 源数据中不存在标注的类别：无。
- 数据许可与再分发限制：以 `dfire_source_license.json` 的原文为准，本文件不替代法律意见。
- 训练期非有限损失处理：策略 `skip`，跳过 15 个批次；被跳过的批次从未参与梯度更新，该数字来自逐轮记录。

**crop_pest**

- 类别覆盖策略（来源 `crop_pest_manifest.json` 的 `class_coverage.policy`）：`explicit_allow_unobserved_ip102`。
- 源数据中不存在任何标注的类别（无法训练、也无法评估）：id [60, 61, 64, 76, 81]，名称 ['Viteus vitifoliae', 'Colomerus vitis', 'Polyphagotars onemus latus', 'Phyllocoptes oleiverus ashmead', 'Parlatoria zizyphus Lucus']。
- 数据许可与再分发限制：以 `ip102_source_license.json` 的原文为准，本文件不替代法律意见。
- 训练期非有限损失处理：策略 `skip`，跳过 256 个批次；被跳过的批次从未参与梯度更新，该数字来自逐轮记录。

以上「不可训练 / 不可评估」的表述来自准备阶段的源侧审计记录，不是对模型能力的推测。

## 附录：证据缺口

无：本报告全部输入均存在且可读。

各任务的 mAP 只对测试划分有效；训练与验证之间的差距、部署现场的域偏移均未评估。
