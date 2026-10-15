# 低空之眼 · 低空巡检决策层软件

**把无人机拍回来的图，变成一张能落地执行的工单。**

低空之眼是一个 B/S 架构的**决策层**软件原型：输入一张巡检图，输出「检出什么缺陷 → 判几级 → 依据哪一条规程 → 派给谁 → 几小时内到 → 谁来复核」，每一步都带条款出处，没有依据时**明确拒答**而不是编一段听起来合理的话。

> 参赛信息：第十八届蓝桥杯全国大学生软件和信息技术大赛 · **数字科技创新赛（非命题赛）**，作品提交截止 2027-04-26 14:00。

---

## 一、它做什么，不做什么

| 做 | 不做 |
| --- | --- |
| 识别：从图像中检出缺陷目标，给出类别、位置、置信度 | **不造无人机、不做飞控、不卖硬件** |
| 诊断：把缺陷对应到规程条款，生成可执行处置动作 | 不训练通用大模型，不做底层算法创新 |
| 分级派单：按确定性规则表定级、定班组、定 SLA | 分级**不交给大模型**判断（规则表在 `config/routes.yaml`） |
| 可溯源：每条动作强制带条款号，可点开看原文 | 不允许无依据生成处置建议 |
| 拒答：知识库没覆盖就说不覆盖，前端分三种判定（检索命中 / 超出覆盖范围拒答 / 按设计拒答） | 不猜、不凑、不用相似条款顶替 |
| 评估：Recall@k / MRR / 拒答率 / 时延 / 三组对照实验 | 不在合成演示图上宣称检测精度（见第七节口径声明）|

---

## 二、当前状态速览（可复现）

**已跑通并实测**（数字都来自仓库内脚本，命令见第五节）：

- 知识库：3 篇文档 → **68 个条款块，带条款号 65 块（95.6%）**；BM25 = rank_bm25，向量 = hashing_tfidf(dim=4096)，重排 = lexical。
- 检索（56 题金标集，默认词法档）：**Recall@1 89% / Recall@3 93% / Recall@5 95%，MRR 0.917**；越界拒答率 **100%**。
- 检索（可选 BGE 档：`bge-m3` 向量 + `bge-reranker-v2-m3` 重排）：**Recall@1 95% / MRR 0.973**，留出题 Recall@1 由 **50% 提到 90%**、拒答率仍为 100%；代价是单次检索从 4 ms 涨到约 2.5 s（CPU 上跑 cross-encoder），详见 7.5 节。
- 时延（本机 CPU，无 GPU）：检索 P50 ≈ 4 ms / P95 ≈ 5.2 ms（n=168）；端到端（4 张演示图）P50 ≈ 6.6 ms / P95 ≈ 24 ms（n=8）。
- 端到端链路：单图 → 检测 → 诊断 → 分级派单 → 工单落盘 → HTML/PDF 报告，四张演示图全部跑通。
- 三组对照实验：可溯源率 基线A 纯检测 **0%** / 基线B 直接问大模型 **0%** / 本系统 **100%**；引用可核验率 **100%**（逐字回查 22 条引用）。
- 测试：`python -m unittest discover -s tests` 本次实跑 **264 项全绿**（20.5 秒，退出码 0）；历史记录为早期 **118 项单元测试全绿**（`unittest`，标准库）、CPLID 接入轮次 **130 项全绿**（见 7.6 节），数量增长来自此后陆续新增的回归用例（巡检管线、GPU 回收、交付守卫与坐标几何契约）。
- 接口：12 个 API 路由 + 3 个静态挂载，自检脚本 13/13 通过。
- Web 控制台：4 个页签（分析台 / 工单 / 知识库 / 追溯），免构建、无第三方运行时依赖、**断网可跑**。

**没做 / 是短板的**（写在最前面，免得被当成已完成）：

1. **默认检测仍是 sidecar 演示，但已有 CPLID 自训练真实权重**。默认演示框读自同名 `.detections.json`，只验证链路连通性，不能用于检测精度结论；独立的 `config/app.cplid_gpu.yaml` 已接入 GPU 自训练的 FCOS ONNX 权重，CPLID 留出集逐类漏检率为缺陷 **0%**、正常绝缘子 **3%**，同时存在误报与重复框，不能外推到其他场景（见 7.6 节）。烟火 / 农作物害虫扩展的数据准备、冒烟门控、24 轮正式训练与独立 ONNX 复评均已完成（见 7.7 节与第十一节第 2 条），逐轮曲线与测试集指标一律以 `docs/proposal/06d_trained_detectors.md` 为准。
2. **默认检索是词法实现，可选 BGE 语义档已验证**。默认 BM25 + hashing_tfidf + 词法重排 + 领域术语表在留出集上 Recall@1 为 **50%**；独立 BGE 档已达到 **90%**，但 CPU 单次检索约 2.5 秒，默认档仍保留零下载与低时延优势（见 7.5 节）。
3. 知识库是**自编占位文档**（3 份演示用规程），不是真实 GB/DL/T 标准原文。
4. 通知（钉钉/飞书）与天气工具默认关闭，需要配置环境变量。

---

## 三、快速开始

前置：Python 3.12（本项目用 3.12.14 验证），Windows PowerShell。

```powershell
# 1) 建虚拟环境并装依赖
cd D:\DeepSeek\gzqu\低空之眼
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 构建知识库（data/raw → 条款切分 → 索引）
.venv\Scripts\python.exe -m loweye.cli.build_kb

# 3) 跑一次单图分析（识别 → 诊断 → 分级 → 派单 → 归档）
.venv\Scripts\python.exe -m loweye.cli.analyze --list
.venv\Scripts\python.exe -m loweye.cli.analyze --image demo_tower_01

# 4) 起服务（改完代码必须重启；--no-reload 是有意的，见第九节）
.venv\Scripts\python.exe -m loweye.cli.serve --host 127.0.0.1 --port 8848 --no-reload
# 浏览器打开 http://127.0.0.1:8848/
```

其它常用命令：

```powershell
# 只检索，不过检测
.venv\Scripts\python.exe -m loweye.cli.search --query "绝缘子自爆怎么处理"

# 跑评测，出报告（out/eval_report.md 与 out/eval_report.json）
.venv\Scripts\python.exe -m loweye.cli.evaluate

# 单元测试（118 项是早期历史统计；当前数量以实际运行输出为准）
.venv\Scripts\python.exe -m unittest discover -s tests -t .

# 接口自检（13 项，服务需已在跑）
.venv\Scripts\python.exe tools\api_smoke.py

# 生成合成演示图与标注
.venv\Scripts\python.exe tools\make_demo_assets.py

# 演示数据复位到零工单（先预演）
.venv\Scripts\python.exe tools\reset_demo.py --dry-run
.venv\Scripts\python.exe tools\reset_demo.py --yes

# 窄屏版式体检（走 CDP 强制视口，量盒模型 + 出图）
.venv\Scripts\python.exe -m tools.probe_layout --url http://127.0.0.1:8848/ --width 390 --height 844 --shot out\mobile_console.png

# 拉 BGE 权重到项目内 models/（默认走 ModelScope，国内快；huggingface.co 直连不通）
.venv\Scripts\python.exe -m tools.fetch_models --source ms

# 量 BGE 在本机的真实开销（加载耗时 / 编码吞吐 / 查询耗时 / 峰值内存）
.venv\Scripts\python.exe -m tools.probe_embedder          # 只测嵌入
.venv\Scripts\python.exe -m tools.probe_embedder --both   # 连重排一起测

# BGE 后端的独立索引与评测（与词法后端分开，互不覆盖）
.venv\Scripts\python.exe -m loweye.cli.build_kb --config config/app.bge.yaml
.venv\Scripts\python.exe -m loweye.cli.evaluate --config config/app.bge.yaml `
  --report out/eval_report_bge.md --json out/eval_report_bge.json
.venv\Scripts\python.exe -m tools.eval_compare --a out/eval_report.json --b out/eval_report_bge.json

# 方案书 PDF 与演示录屏（都是确定性重跑，不靠人手操作）
.venv\Scripts\python.exe -m tools.build_proposal          # → out/低空之眼-参赛方案书.pdf
.venv\Scripts\python.exe -m tools.record_demo --url http://127.0.0.1:8848/ --out out/demo_screencast.mp4
```

### 在手机上打开（局域网访问）

服务默认只绑回环地址，手机连不上**不是因为防火墙**，而是局域网地址上根本没有监听。三步：

```powershell
# 1) 绑到所有网卡，局域网地址才会有监听
.venv\Scripts\python.exe -m loweye.cli.serve --host 0.0.0.0 --port 8848 --no-reload

# 2) 查本机局域网地址（忽略 169.254.* 链路本地地址）
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.254.*' }

# 3) 放行入站端口（需管理员权限；下面按热点网段收窄，不要对全网开放）
New-NetFirewallRule -DisplayName 'loweye 8848 LAN demo' -Direction Inbound -Protocol TCP `
  -LocalPort 8848 -Action Allow -Profile Any -RemoteAddress 192.168.43.0/24
```

手机与电脑连同一个 Wi-Fi（或电脑连手机热点），浏览器打开 `http://<第 2 步的 IPv4>:8848/`，
例如 `http://192.168.43.4:8848/`。**服务没有任何鉴权**，只应在自己控制的网络里开；
演示结束用 `Remove-NetFirewallRule -DisplayName 'loweye 8848 LAN demo'` 收回放行。

界面已做窄屏适配（`@media (max-width: 720px)`），手机上是可用版式而不是把桌面版硬缩。

### 给老师 / 评委看：一个能直接发出去的公网地址

`192.168.x.x` 这类局域网地址只有同网段能打开，发给别人是打不开的。要一个**能直接发出去**的
地址，用 Cloudflare 免登录快速隧道（不需要账号，只要能出网）：

```powershell
# 1) 下载一次（约 53 MB，放在项目内）
Invoke-WebRequest 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' `
  -OutFile 'tools\bin\cloudflared.exe'

# 2) 服务保持运行，另开一个终端起隧道
tools\bin\cloudflared.exe tunnel --url http://127.0.0.1:8848 --no-autoupdate
# 输出里会给一个 https://xxx-xxx-xxx.trycloudflare.com 地址，直接发给对方即可
```

三点必须知道：① 免账号隧道**没有可用性承诺**，且地址每次启动都会变；
② 本机要保持开机、隧道进程不能关；③ 服务**没有任何鉴权**，拿到地址的人都能跑分析、
读演示知识库——演示结束后在隧道终端按 `Ctrl+C` 关掉即可收回。

---

## 四、目录结构

```
低空之眼/
├─ config/                 # 全部阈值与规则，代码里不写魔法数字
│  ├─ app.yaml             #   全局配置（检索/检测/生成/工具/评测）
│  ├─ routes.yaml          #   分级派单规则表（确定性，不交给大模型）
│  ├─ lexicon.yaml         #   领域术语表（47 条，口语 → 规程术语）
│  └─ report_template.html #   Jinja2 报告模板（供 PyMuPDF Story 排版）
├─ data/
│  ├─ raw/                 # 3 份演示知识文档（自编占位文本）
│  ├─ processed/           # 切分产物 clauses.jsonl
│  ├─ index/               # 检索索引（向量 + 元数据）
│  └─ assets/              # 4 张合成演示图 + 同名 .detections.json
├─ eval/gold_qa.yaml       # 金标问答集：34 原词 + 12 改写(开发集) + 10 改写(留出集) + 8 越界
├─ loweye/                 # 后端源码（40 个 py 文件，4324 行）
│  ├─ parsing/             # 文档加载与「章-节-条」语义切分
│  ├─ retrieval/           # BM25 + 向量 + RRF 融合 + 重排 + 术语表
│  ├─ detect/              # 检测抽象、SAHI 切片、ONNX 推理、sidecar 回放
│  ├─ llm/                 # 生成后端：template（默认）/ openai_compat
│  ├─ agent/               # 编排（显式状态机）、分级派单、五个零成本工具
│  ├─ eval/                # 金标集、指标（Recall/MRR/mAP）、三组对照基线
│  ├─ cli/                 # build_kb / analyze / search / serve / evaluate
│  ├─ api.py               # FastAPI 路由 + 静态挂载
│  └─ schemas.py           # 全部结构化契约（Pydantic v2）
├─ web/                    # 前端（3 个文件 1958 行，免构建、断网可跑）
│  ├─ index.html / styles.css / app.js
│  └─ vendor/icons.svg     # 本地 Lucide 图标 sprite（43 个，不依赖 CDN）
├─ tools/                  # 开发与演示脚本（19 个，2491 行）
├─ tests/                  # 单元测试（早期历史统计：9 个文件，778 行，118 项）
└─ out/                    # 报告、评测产物、截图
```

---

## 五、数据流与六个模块

```
图片 ──▶ M1 识别 ──▶ M2 检索(诊断) ──▶ M3 编排(分级·派单) ──▶ M4 报告/工单 ──▶ M5 Web 控制台
           │              │                    │                    │
           └──────────────┴────────────────────┴────────────────────┘
                                  M6 评估与观测（贯穿）
```

| 模块 | 做什么 | 关键实现 |
| --- | --- | --- |
| M1 识别 | 检出缺陷目标 | `detect/`：sidecar 回放（演示）/ ONNX（真实）/ null；SAHI 切片 + 重叠 + 跨切片 NMS 去重 |
| M2 诊断 | 找依据、给动作 | `parsing/` 切分 → `retrieval/` 两路召回 → RRF 融合 → 重排 → 术语表扩展；动作句只从条款原文里抽，抽不到就退回列引用；条件分支与场景范围两道闸门在这里生效 |
| M3 编排 | 显式状态机 | `agent/pipeline.py`：识别 → 诊断 → 分级 → 派单 → 归档，不用 LangChain，状态与降级链写在代码里 |
| M4 派单与报告 | 工单 + PDF | `agent/tools.py`：工单落盘 JSONL、Jinja2 + PyMuPDF 出 HTML/PDF、通知与天气（可选） |
| M5 控制台 | 给人看 | `web/`：四个页签、检测框叠加、四路分数条、条款原文弹层、工单复核 |
| M6 评估 | 证明有效 | `eval/` + `cli/evaluate.py`：Recall@k、MRR、拒答率、时延 P50/P95、三组对照实验 |

**降级链**（每一环失败都有明确退路，且退路会写进结果的 `degradations`，不静默）：

| 环节 | 首选 | 失败时 |
| --- | --- | --- |
| 检测 | ONNX + 切片 | 缺权重/缺 onnxruntime → sidecar 回放 → 空检测器（原因进 `degradations`）|
| 向量 | sentence-transformers(BGE-M3) | 未安装 → hashing_tfidf（零下载、确定性）|
| 重排 | cross-encoder(bge-reranker-v2-m3) | 未安装 → 词法重排（绝对分不随批次归一化）|
| 生成 | 云端 OpenAI 兼容接口 | 超时/失败 → template 后端（从条款确定性拼装，结构上不可能编造条款号）|
| 检索 | 向量 + BM25 | 向量库不可用 → 仅 BM25 |

**范围控制（三道闸门，都是为了「宁可少答，不可错答」）**：

| 闸门 | 拦什么 | 例子 |
| --- | --- | --- |
| 场景文档范围 | 各专业规程只在本场景内被引用 | 输电缺陷不会引用《植保作业指引》的通用报告条款（`retrieval.scene_docs`）|
| 条件分支校验 | 条件说的不是当前缺陷时，该句不抽为动作 | 金具锈蚀不照搬「出现裂纹、变形或磨损的，应按危急缺陷流程处理，立即申请停电更换」——那会是一次不必要的停电；被跳过的条件会在结论里点名 |
| 参考条款 | 命中但没成为动作的条款（判定句、响应分级）仍保留 | 火情工单的依据是 3.1.1／3.2.3，而「一级／二级响应」判据以 `role=reference` 呈现，让人能核对 P1 是怎么判出来的 |

三条都不影响知识库页的自由检索：那是全库检索，闸门只在「生成处置动作」这条链上生效。

---

## 六、接口

12 个 API 路由 + 3 个静态挂载（`/` 前端、`/assets` 演示图、`/uploads` 上传件；`/` 是通配前缀，**必须最后挂载**）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康与后端清单（含降级原因，失败不 500 而是 `status=degraded`）|
| GET | `/api/kb/stats` | 知识库统计 + **覆盖阈值 min_relevance** + 术语表条数 |
| GET | `/api/assets` | 演示图列表（含是否有标注）|
| POST | `/api/search` | 条款检索（空查询 422；返回 `search_query`、`matched_terms`、`scope_excluded`、四路分数）|
| POST | `/api/analyze` | 按 `image_id` 跑全链路 |
| POST | `/api/analyze/upload` | 上传图片跑全链路 |
| GET | `/api/traces` `/api/traces/{id}` | 追溯列表与单条快照 |
| GET | `/api/orders` | 工单列表（`{"items":[...],"count":N}`）|
| POST | `/api/orders/{id}/review` | 人工复核（确认派单 / 退回）|
| GET | `/api/report/{trace_id}?fmt=html\|pdf` | 报告（非法 fmt → 400）|
| POST | `/api/selfcheck` | 六步全链路自检 |

交互式文档：服务起来后访问 `http://127.0.0.1:8848/docs`。

---

## 七、评估：方法与结果

金标集 `eval/gold_qa.yaml` 共 **64 题**：34 道原词题、12 道改写题（口语化、不含文档原词，**开发集**）、10 道改写题（术语表冻结之后才写，**留出集**）、8 道越界题。判定要求 `(文档, 条款号)` 双匹配——条款号跨文档会重号，只匹配条款号会自欺。

### 7.1 检索（五种配置对照，全量 56 道域内题）

| 配置 | MRR | 说明 |
| --- | --- | --- |
| 仅 BM25 | 0.861 | 关键词字面命中 |
| 仅向量 | 0.898 | hashing_tfidf 语义相似（弱语义）|
| RRF 融合、不重排 | 0.893 | 两路排名融合 |
| **本系统（融合 + 重排 + 术语表）** | **0.919** | 最终排序 |
| 融合 + 重排，**关闭术语表** | 0.818 | 术语表的单独贡献 |

分组看，差距比总分更有信息量：

| 分组 | 题数 | 本系统 Recall@1 | 关闭术语表 Recall@1 |
| --- | --- | --- | --- |
| 原词题 literal | 34 | **100%** | 100% |
| 改写题 paraphrase（开发集）| 12 | **92%** | 42% |
| 改写题 heldout（留出集）| 10 | **50%** | — |

**这三行数字是本项目最重要的诚实记录**：术语表把开发集从 42% 拉到 92%，但换一批没见过的口语表达，命中率掉到 **50%**。我们**没有**回头去改术语表把留出集刷上去——那等于把留出集变成开发集，指标就失去意义。正确结论是：词法路线的泛化能力到此为止，要换语义向量（BGE-M3）+ cross-encoder 重排。

另一个实测结论：改写题里正确条款与错误条款的**绝对分区间是重叠的**（0.135 ~ 0.362），不存在一个能把两者分开的阈值。所以遇到「正确条款但分数低于阈值」时，本项目的选择是**拒答**而不是降阈值硬答——宁可少答，不可错答。

### 7.2 拒答能力

8 道越界题拒答率 **100%**。其中 4 道边界题（红外测温、直升机巡检、农药配比、带电作业）的最相似条款**恰好是文档自己的范围排除声明**（「本导则不适用于……」）——实测分数 0.2629 / 0.4881 / 0.3712 / 0.5342，都高于阈值 0.18，靠阈值拦不住。为此系统区分两种拒答机制，前端给出三种判定文案（`web/app.js:722`）：

- `scope_excluded = true` → 「超出覆盖范围 拒答」：最相似的那条本身就是排除声明；
- 最高分低于阈值 → 「按设计拒答」：知识库确实没有覆盖；
- `covered = true` → 「检索命中」：正常作答。

三种判定分开显示，因为它们对用户的含义完全不同。

### 7.3 时延与对照实验

| 指标 | 实测（CPU，无 GPU）|
| --- | --- |
| 检索 P50 / P95 | 4.04 ms / 5.17 ms（n=168，最大 6.34；同一台机器多次跑在 4.0~4.2 / 5.2~5.9 之间）|
| 端到端 P50 / P95 | 6.62 ms / 24.4 ms（n=8，最大 24.4；早前几轮 P95 曾在 24~31 之间）|
| 分阶段 P95 | 检测 18.33 ms、诊断 9.26 ms、分级派单 1.18 ms、归档 0.0 ms |

三组对照（同一批演示图、同一分母，每个「图片 × 缺陷类别」算一个样本）：

| 组 | 可执行动作比例 | 可溯源率 | 引用可核验率 |
| --- | --- | --- | --- |
| A 纯检测（只报框）| 0% | 0% | — |
| B 直接问多模态大模型（无知识库）| 100% | **0%** | — |
| C 本系统 | 100% | **100%** | **100%**（逐字回查全部引用，28 条，含参考条款）|

「可执行率」三项都用同一个判据（`is_actionable`，对三组一视同仁），所以**它不构成质量优势**——B 组说的话同样"像建议"。真正的差别在**可溯源率**与**引用可核验率**：A、B 结构上就做不到，不是做得差。

### 7.4 口径声明（引用本仓库数字前必读）

1. 演示图是 **Pillow 程序化合成图**，标注是同目录 sidecar，**不能用于任何检测精度结论**。`eval_detection_selfcheck()` 里的 mAP 是「用同一份标注自证」的循环，脚本与报告均已标注，禁止对外引用。
2. 知识库是**自编占位文档**，条款内容为演示而写，不是任何标准原文。检索指标反映的是**检索策略**的相对差异，不代表在真实规程上的表现。
3. 基线 B 默认走**脚本化代理**（`arm_b_mode: scripted`），离线可复现；它代表「一个不做检索直答的模型」的**结构特征**（无引用、无溯源），**不代表某个具体模型的真实水平**。切换 `--arm-b api` 并配置 `llm.base_url` 后可跑真实 API。
4. 术语表在 paraphrase 组上开发，因此该组只能算开发集结果。

### 7.5 换 BGE 后端后的复测（可选质量档）

默认档是词法后端：零下载、检索 P50 约 4 ms、内存占用小，现场演示稳。另有一套 **BGE 档**：向量换成 `BAAI/bge-m3`（1024 维），重排换成 `BAAI/bge-reranker-v2-m3`，权重放在 `models/` 下，配置独立在 `config/app.bge.yaml`，索引独立在 `data/index_bge`，与默认档互不覆盖。

同一套 56 题金标集、同一台机器、同一份代码，两档对照（完整对照表见 `out/eval_compare.md`，可由 `python -m tools.eval_compare` 一键重算）：

| 指标 | 词法档（默认） | BGE 档 |
| --- | --- | --- |
| 全量域内题 Recall@1 / MRR | 89% / 0.917 | **95% / 0.973** |
| 字面题（n=34） | 100% / 1.000 | 100% / 1.000 |
| 改写题·开发集（n=12） | 92% / 0.926 | 83% / 0.917 |
| **留出题（n=10，从未参与调参）** | **50% / 0.625** | **90% / 0.950** |
| 越界拒答率（8 题） | 100% | 100% |
| 检索 P50 / P95（n=168） | 4.1 / 5.08 ms | 2524 / 2956 ms |
| 端到端 P50 / P95（n=8） | 9.18 / 25.56 ms | 3864 / 5202 ms |

三件事必须一起说清楚：

1. **留出集从 50% 提到 90%**，达到了「留出集 Recall@1 ≥ 80%」这个里程碑；改造后仍有 1 题答错，不是满分。
2. **改写题（开发集）从 92% 掉到 83%**：这一组是术语表的开发集，词法档在这组上是「练过的」，BGE 档没有针对它调参。两组数字放一起看，才不会被单组数字误导。
3. **代价是时延**：cross-encoder 在 CPU 上给每题 20 个候选打分，单次检索约 2.5 秒（词法档 4 毫秒）。BGE 档适合离线批处理与答辩对照，不适合现场逐张实拍演示；真要用在产线，需要 GPU 或更小的重排模型。

这一轮改造顺带修掉两个会让「拒答」静默失效的缺陷，都留了代码注释：

- **交叉编码器被套了两层 sigmoid**：`sentence-transformers` 的 `CrossEncoder` 在 `num_labels=1` 时默认已经过一层 Sigmoid，代码里再套一层，分数被压到 0.5 附近（域内 0.51~0.73、域外 0.50~0.66，两组几乎重叠），阈值拒答直接失效——拒答率一度从 100% 掉到 38%。现在显式用 `activation_fn=torch.nn.Identity()`，只由本项目套一层。
- **范围排除闸门判在重排后的第一名**：BGE 重排会把「本导则不适用于……」这类排除声明从第 1 名挤到第 2、3 名，闸门就失效了。现在闸门改判**召回阶段（BM25 与向量的并集）前 3 名**里有没有排除声明——召回衡量的是主题相近，重排衡量的是「这条能不能当答案」，两者不是一回事。

复现命令见第三节；原始报告在 `out/eval_report_bge.md` 与 `out/eval_report_bge.json`。

### 7.6 检测环节：真实权重通道与逐类漏检率

此前检测后端读的是同名 `.detections.json`（sidecar 合成标注），只能证明链路连得通。现在真实权重通道也接上了，并留下第一批可复现的检测数字——**其中包含一个必须直面的负面结果**。

| 项 | 内容 |
| --- | --- |
| 推理后端 | `onnxruntime`（CPU）+ 既有 SAHI 切片（640 瓦片、overlap 0.2）+ 跨片 NMS |
| 权重 | `models/detector/rtmdet-tiny-coco.onnx`（21.3 MB），OpenMMLab **RTMDet-tiny**，**Apache-2.0** |
| 为什么不是 YOLOv8/v5 | 公开的「绝缘子权重」几乎都是 Ultralytics 系产物，继承 **AGPL-3.0**，与许可要求冲突 |
| 配置 | `config/app.onnx.yaml`（独立文件，不动默认的 sidecar 配置） |
| 格式适配 | MMDeploy 系是双输出 `dets(1,N,5)` + `labels(1,N)`，与 YOLO 单输出不同；`onnx_detector.py` 现按输出名自动判定布局 |

阳性对照：真实照片（960×686）上检出 `cat` 两个框（0.651 / 0.630，534 ms，4 切片）；合成演示图上 0 检出——因为 COCO 80 类里没有「输电杆塔」。前者证明通道真在检测，后者说明 0 检出不代表链路坏了。

域内实测（CPLID 子集，1152×864 真实航拍图，`tools/eval_detection.py` 逐类统计）：

| 数据集 | 类别 | GT 框 | 命中 | 逐类漏检率 | **类无关召回** | 空图率 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 缺陷图 50 张 | `defect` | 50 | 0 | 100% | **0 / 50 = 0%** | 74% | 428 ms |
| 正常图 50 张 | `insulator` | 101 | 0 | 100% | **0 / 101 = 0%** | 76% | 494 ms |

模型在这 100 张图上只输出了 42 个框，类别是 person / airplane / clock / bed / bench / umbrella / truck / car / bicycle / toilet / traffic light——**一个绝缘子都没框到**。逐类漏检率 100% 是必然的（COCO 没有绝缘子类，标签对不上就不算命中），真正有信息量的是**类无关召回 0%**：忽略类别，只问有没有任何一个框压到绝缘子上，答案是一个都没有。

**自训练这一步也已经做掉**：COCO 零召回证明「必须自训练」之后，在租用的单卡 RTX 4090D 上用 `torchvision` 的 `fcos_resnet50_fpn`（BSD-3-Clause，anchor-free 单阶段，与 RTMDet 同范式）微调了 CPLID 两个类别。为什么不是 mmdet/RTMDet：预编译 `mmcv` 只到 cu121/torch2.2，租用机是 torch 2.8+cu128 且无 `nvcc`，装它要降级重装 torch，风险大于收益（详见 `docs/proposal/06c_detection.md`）。

训练产物 `models/detector/cplid_fcos_1024.onnx`（122.9 MB），30 轮 / batch 6 / 1024 输入 / 约 36.4 s·轮，留出集与检索两档用同一批图（`out/cplid_split.json` 的 val）：

| 留出集 | 类别 | GT 框 | 命中 | **逐类漏检率** | **类无关召回** | 空图率 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 缺陷图 48 张 | `defect` | 48 | 48 | **0%** | **100%** | 0% | 916 ms |
| 正常图 63 张 | `insulator` | 113 | 110 | **3%** | **97%** | 0% | 1081 ms |

**必须同时看的误报**：缺陷集除 48 个命中框外还有 16 个 `defect` 误报、122 个 `insulator` 重复框；正常集有 97 个 `insulator` 重复框、3 个 `defect` 误报。结论是**召回强、精度一般**——工程上靠「正常类不出工单」和「低置信度转人工」两道闸门兜底，而不是假装误报不存在。

接回流水线时修掉三处「不会报错、只会算错」的问题，并在 `config/routes.yaml` 里补上真实权重的两个类别，否则检出的一切都会走默认等级、全部转人工（现场就是「检出 0.82 也判 P3」）：

| 类别 | 规则 | 语义 |
| --- | --- | --- |
| `defect` | **P1 绝缘子破损**、输电运检班、SLA 24 h、强制人工确认 | 缺陷成单，链路最短 |
| `insulator` | **P4 正常**、`normal: true` | 出结论、**不出工单**、不检索条款（否则把「缺陷处置」条款套到正常设备上就是伪依据） |

端到端实测（`config/app.cplid_gpu.yaml`、端口 8851，上传真实航拍图）：缺陷图 → `defect 0.825` → 诊断「绝缘子破损 P1（5 条依据）」→ 1 张工单（输电运检班 / SLA 24 h / 人工复核）；三张正常图 → 全部「绝缘子（未见缺陷）P4」→ 0 张工单。【历史实测记录】该 CPLID 接入轮次单测同步扩到 130 项全绿；本次文档同步未重跑全量测试。

复现：

```powershell
# 阳性对照（应检出 cat）
.venv\Scripts\python.exe -m tools.probe_detector --config config/app.onnx.yaml --image models/detector/samples/sample_cat2.jpg

# 拉 CPLID 子集（HF 镜像；仓库无 LICENSE，仅本地评测，不随作品分发）
.venv\Scripts\python.exe -m tools.fetch_cplid --limit 50

# 逐类漏检率（COCO 权重的负面结果）
.venv\Scripts\python.exe -m tools.eval_detection --config config/app.onnx.yaml `
  --images data/external/cplid/Defective_Insulators/images `
  --labels data/external/cplid/Defective_Insulators/labels/defect `
  --out out/eval_detection_coco_on_cplid_defect.md

# 自训练权重（留出集，与检索两档同一划分）
.venv\Scripts\python.exe -m tools.eval_detection --config config/app.cplid_gpu.yaml `
  --images data/external/cplid/Defective_Insulators/images `
  --labels data/external/cplid/Defective_Insulators/labels/defect `
  --only-from out/cplid_split.json --split-key val `
  --out out/eval_detection_fcos_defect.md --json out/eval_detection_fcos_defect.json

# 起真权重的服务（默认档仍是 sidecar，端口不冲突）
.venv\Scripts\python.exe -m loweye.cli.serve --config config/app.cplid_gpu.yaml --host 127.0.0.1 --port 8851
```

### 7.7 烟火 / 农作物害虫扩展：数据、训练与评测记录

此处“烟火”指 **烟雾 / 火焰**，不含烟花。已有 CPLID 权重的实测结果不代表这些新增场景。

本节只记录**稳定的数据事实**（来源、体积、哈希、划分、隔离）；逐轮曲线、权重选型与测试集指标由 `tools/write_model_report.py` 从落盘制品确定性生成到 `docs/proposal/06d_trained_detectors.md`，**不在 README 内重复数字**，避免两处随训练推进而漂移。两个模型的训练精度、权重选型与测试集指标一律以该报告为准；报告已生成，本仓库不再另作结论。

| 数据源 | 已验证状态（含历史失败记录） | 证据 |
| --- | --- | --- |
| IP102 检测子集 | 下载哈希核验通过：**807750864 字节**；SHA-256：`38c8a26152baeea9283d593a3ab955e8fe3258105d08a0e30372765111f57fb0` | `out/inspection_gpu/ip102_transport.json`、`out/inspection_gpu/ip102_source_license.json` |
| IP102 首次准备（历史） | 当时**失败 180 项**：174 项源数据使用 0 基类号（初版转换按 1..102 校验而拒绝）、5 项缺标注、1 项拼接 XML；该轮未通过准备，后续轮次已修复（见下面「IP102 准备通过」行）；哈希通过不等于准备或训练完成 | `out/inspection_gpu/ip102_initial_preparation_errors.json` |
| IP102 第二次准备（历史） | 当时仍失败：**7 项 XML / TXT 双标注冲突**；已隔离首次发现的 **6 项源问题**（5 缺标注 + 1 拼接 XML），该轮尚不可用于训练 | `out/inspection_gpu/ip102_second_preparation_errors.json` |
| IP102 第三次准备与原源审计（历史） | 第三次因缺少前景类号 **60 / 61 / 64 / 76 / 81** 未通过；隔离累计 13 项。隔离 / 去重前的全部源标注审计：XML **18976 份、有效 18975 份**，TXT **7 份**；合并统计原源本就缺少 0 基类号 **59 / 60 / 63 / 75 / 80**，不是隔离或去重造成丢类 | `out/inspection_gpu/ip102_third_preparation_errors.json`、`out/inspection_gpu/ip102_source_class_audit.json` |
| IP102 准备通过 | 显式允许源缺类 + 隔离坏标注后通过：**102 类表保留、train 12894 / val 2276 / test 3798（官方 test 保留）**，`voc_class_id_base=0` 映射到内部 1..102，隔离 13 项，无跨划分泄漏 | 运行时清单 `out/inspection_gpu/crop_pest_manifest.json`；划分口径 `split_strategy` 字段 |
| D-Fire 下载与解包 | 已完成：**3049605157 字节**；MD5 `9745236834…`（与源站 ETag 一致）；SHA-256 `65d81b60…`；解包 **43055 项 / 3118334483 字节** | `out/inspection_gpu/dfire_transport.json`、`out/inspection_gpu/dfire_source_license.json` |
| D-Fire 固定划分 | 严格模式首跑报 **340 项**越界 / 非有限框；显式隔离后通过：**train 13898 / val 3045 / test 4244**，类别 `smoke` / `fire`，无跨划分泄漏 | `out/inspection_gpu/dfire_strict_preparation_errors.json`、`out/inspection_gpu/dfire_preparation_probe.json` |

实施调整为显式允许源数据缺类并**保留 102 类别表**；隔离 13 项坏标注（含 7 项 XML / TXT 双标注冲突）后准备通过。但**保留类别表不等于缺失类可训练**：这 **5 类（0 基类号 59 / 60 / 63 / 75 / 80）在源数据里没有样本，无法训练也无法评估**，因此不能宣称模型具备全部 102 类识别能力；测试集上的逐类指标对它们只能标为不可评估。原源审计计数包含双标注，不是去重后的独立目标数。

**许可与适用边界必须分开披露**：

- `torchvision` / FCOS 的 **BSD-3-Clause 是实现许可**，不代表训练数据或派生权重可分发；各数据源与权重仍需分别核对授权。
- **IP102** 官方来源为 [xpwu95/IP102](https://github.com/xpwu95/IP102)：学术用途免费，其他用途须联系作者；镜像宣称的 PDDL 不覆盖官方条款，**竞赛 / 商业用途及数据、权重分发许可尚未确立**。本项目不再分发原始数据。它是农作物害虫数据，**不是植物病害数据，也未完成航拍验证**。来源工具为 `tools/fetch_ip102.py`，许可记录为 `out/inspection_gpu/ip102_source_license.json`。
- **D-Fire** 官方来源为 [DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset)：集合标示 **CC0-1.0**，但上游不主张拥有全部原始图像权利，不能将集合许可解读为全部原图权利已清理；本项目不再分发原图。其类别为烟雾 / 火焰，**不包含烟花标注，未完成航拍验证**。来源与披露见 `tools/fetch_dfire.py`。

### 7.8 扩展模型的部署配置

两个扩展检测器各带一份**变体配置**，由 `tools/write_detector_variant_config.py` 以 `config/app.cplid_gpu.yaml` 为模板、按训练元数据（`*_fcos_metadata.json` 的类别顺序与 `preprocessing.input` 边长）确定性生成，不手改数字；`--check` 可对已存在的文件复算比对，不一致即报错。

| 变体配置 | 检测器 | 输入边长 | 类别数 | 服务端口 | `model_version` |
| --- | --- | --- | --- | --- | --- |
| `config/app.crop_pest_gpu.yaml` | ONNX | 800 | 102 | 8853 | `crop-pest-fcos-r50fpn-800` |
| `config/app.fire_smoke_gpu.yaml` | ONNX | 640 | 2 | 8852 | `fire-smoke-fcos-r50fpn-640` |

三处**必须落在同一份配置里**，写错都不会在启动时报错，只会在推理期出错或标错版本：

- `detect.onnx_path`：回收后 ONNX 的落点是 `models/detector/`（回收由 SHA-256 校验后才落位）。
- `detect.input_size`：**必须与导出图的常量边长一致**；写成别的边长不会在构造期报错，只会在推理期抛 `InvalidArgument: Got invalid dimensions for input: images … index: 2`（已实测：把 1024 喂给 800 的图即此报错，且输出为 `detections: null`）。
- `detect.model_version`：独立字段，漏改会让界面把 crop_pest 的结果标成 CPLID 的版本号。

部署自检（缺省用程序化合成图，不需要真实照片；制品回收前可用 `--onnx` 指向任意已落盘的导出图先自检）：

```powershell
.venv\Scripts\python.exe -m tools.check_detector_config --config config/app.crop_pest_gpu.yaml `
  --json-out out/inspection_gpu/check_crop_pest_config.json
```

退出码：`0` 配置生效（`resolved_backend: onnx`）；`3` 静默降级为 sidecar（`loweye/detect/base.py` 在 ONNX 后端构造失败时不抛错）或推理报错——对真实照片表现为零检出，必须处理；`2` 配置或参数本身有问题。

复算已生成的配置是否与参数一致（`--check` 不写文件；参数须与原生成一致，`--title` 即 `app.subtitle`，写错会报 `identical: false`）：

```powershell
.venv\Scripts\python.exe -m tools.write_detector_variant_config `
  --task crop_pest --onnx models/detector/crop_pest_fcos_800.onnx `
  --metadata out/inspection_gpu/crop_pest_fcos_metadata.json `
  --model-version crop-pest-fcos-r50fpn-800 --port 8853 `
  --title "农作物害虫检测（IP102 / FCOS-R50-FPN 800）" `
  --out-file config/app.crop_pest_gpu.yaml --check
```

起服务：

```powershell
.venv\Scripts\python.exe -m loweye.cli.serve --config config/app.crop_pest_gpu.yaml --host 127.0.0.1 --port 8853 --no-reload
```

类别表保留 102 类，但**其中 5 类在源数据里没有样本**（0 基类号 59 / 60 / 63 / 75 / 80），模型既无法识别也无法评估它们；原图不随仓库分发（见 7.7 节与第十二节）。

---

## 八、演示流程（3 分钟）

演示前 30 秒准备：

```powershell
.venv\Scripts\python.exe tools\reset_demo.py --yes     # 回到零工单
.venv\Scripts\python.exe -m loweye.cli.serve --host 127.0.0.1 --port 8848 --no-reload
```

1. **分析台**：点 `demo_tower_01` → 检测框出现（红框 P1 绝缘子自爆 0.93、蓝框 P2 金具锈蚀 0.81）→ 四个节点按真实耗时依次点亮 → 诊断卡给出 4 条动作，每条带条款号。
2. **点条款 chip**：弹出条款原文，逐字对得上；下方「参考条款」是命中但没成为动作的条款（如判定句 2.1.1）。
3. **讲一个「没答」的例子**：金具锈蚀这张卡里没有「立即申请停电更换」——因为那句只适用于裂纹/变形/磨损，系统把条件分支挡掉了。这比多答一条更能说明工单能不能用。
4. **工单页**：P1 工单显示「待人工确认」及原因（P1 属强制人工确认范围），点「确认派单」。
5. **知识库页**：搜「红外测温的温升判据是多少」→ **范围拒答卡**（最近条款恰好是排除声明）；再搜「瓷瓶上落了厚厚一层灰 算几级」→ 术语条显示「识别到的领域术语 绝缘子·污秽」，命中 2.2.2。
6. **追溯页**：下载刚生成的 HTML/PDF 报告，展示「降级项 / 分阶段耗时 / 依据条款与参考条款」三节。

深链（录屏与分享用）：`/?image=demo_tower_01`、`?view=kb&q=绝缘子自爆`、`?view=orders`、`?view=traces`。

---

## 九、工程约定与踩过的坑

- **改完后端代码必须重启服务**：`--reload` 在本机不可靠（改 `schemas.py` 后接口仍按旧契约校验），所以脚本默认 `--no-reload`。
- **`/` 静态挂载必须最后挂**：Starlette 按注册顺序匹配，先挂 `/` 会把 `/assets/*` 全截走。
- **改代码前先读目标区域**：本仓库注释写得细，凭记忆写 `old_string` 会连续 edit 失败。
- **`with_suffix(".tmp.pdf")` 会抛 `ValueError`**（不接受多点后缀），用 `with_name(stem + ".tmp.pdf")`；对 `DocumentWriter` 刚写完的文件做原地替换在 Windows 上会 `PermissionError`，必须先写临时名再落最终名。
- **PDF 体积**：直出 3.6 MB（中文回退字体整包嵌入），用 `save(garbage=4, deflate=True, clean=True)` 压到 1.7 MB。
- **前端 `[hidden]` 要显式兜底**：自定义 `display: grid` 会盖掉浏览器默认的 `[hidden]{display:none}`，导致弹层加载即张开（已加 `[hidden]{display:none!important}`）。
- **headless 截图必须加 `--force-prefers-reduced-motion`**：否则定格在入场动画中途，整页半透明，极易误判成渲染 bug。
- **`--window-size` 小于约 500px 时布局不会真的变窄**：Windows 上浏览器窗口有最小宽度，`--window-size=390,844` 的 `--screenshot` 只是把左上角 390×844 裁出来，布局仍按 ~504px 计算——被切掉的元素看起来"正常"，窄屏溢出被完全掩盖。要可信的窄屏结论用 `tools/probe_layout.py`：它走 CDP `Emulation.setDeviceMetricsOverride` 强制视口，同时吐 `getBoundingClientRect` 与 `Page.captureScreenshot`。
- **窄屏断点覆盖必须写在样式表末尾**：`.pipeline` 等基础规则在 `styles.css` 中部，同优先级下后置声明获胜，`@media (max-width: 720px)` 若写在前面会被静默盖掉（表现为「改了没反应」）。
- **PowerShell 5.1 的多行 `python -c`**：here-string 里含 `$` 插值会被吞，含 `$` 的脚本请写成 `.py` 文件再跑；jieba 首次加载会往 stderr 打 "Building prefix dict"，PowerShell 把它渲染成 `NativeCommandError` 并让管道返回退出码 1，属噪声不是失败。
- **单字情态词必须带前后顾断言**：`应` 要挡「响应／供应／相应／应用／应对」，`按` 要限定成「按…执行／办理／规定」这类口径。否则「二级响应：……」「按季度盘点」都会被当成处置动作抽进工单（已各留一条回归测试）。
- **条件分支的引导动词要含后果类**：「异物**已导致**放电痕迹或灼伤的，应立即申请停电清除」里的条件不是由「出现／发现」引出的，只认前半类动词会漏判，于是后果分支被当成当前动作（已加 `导致|造成|引发|产生`）。

---

## 十、前端规范（本项目的自定约束）

白（主）/ 黑 / 蓝（辅），红色只用于 P1 与退回操作；每个元素都有出现、过渡、消失三段动画；视图切换优先走 `document.startViewTransition()` 做跨帧形变，不支持的浏览器退化为关键帧；微交互包含按钮 hover 抬升 + `:active` 缩放、图标 hover 旋转、页签下划线形变滑动、徽标弹跳、数字滚动、分数条宽度过渡；全部动效受 `prefers-reduced-motion` 抑制；图标用本地 Lucide sprite（43 个），**不使用 emoji**、不依赖 CDN（断网可用）。

窄屏（≤720px，见 `styles.css` 末尾的断点块）另有一套覆盖：顶栏换行、状态文案 `text-overflow: ellipsis` 省略、自检按钮收成纯图标（`title`/`aria-label` 承担可读名）、页签横向滚动、左栏（演示图库 / 运行后端）下移到识别舞台之后、流水线与统计格改两列、工单卡片操作按钮换行右对齐、提示条改为贴底通栏。改动效果用 `tools/probe_layout.py` 复核，判据是 `documentElement.scrollWidth == innerWidth`（无横向溢出）。

---

## 十一、下一步（按优先级）

1. ~~**换检索后端**~~（**已完成**，见 7.5 节）：`BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` 已接上，留出集 Recall@1 从 50% 提到 90%、拒答率保持 100%；代价是 CPU 上单次检索约 2.5 s。下一步若要把它用在产线，需要 GPU，或换更小的重排模型 / 量化 ONNX。
2. **改进已有真实检测权重并扩展场景**：CPLID 的 GPU 自训练、ONNX 接入与留出集评估已完成（见 7.6 节），不再等待全部自训练；现有 FCOS 权重缺陷漏检率 **0%**、正常绝缘子漏检率 **3%**，但误报与重复框仍需改进，并需独立数据验证泛化。COCO 权重类无关召回 **0%** 是先前的负面对照，不是当前自训练权重的表现。
   **烟火 / 农作物害虫扩展**：两个数据集均已完成下载哈希核验与固定划分（IP102 保留官方 test 划分；D-Fire 隔离 340 项越界或非有限框后通过），两个模型的冒烟门控均已通过，正式训练（各 24 轮）与独立 ONNX 复评均已完成（见 7.7 节与 `docs/proposal/06d_trained_detectors.md`）。逐轮曲线、权重选型与测试集指标一律以 `docs/proposal/06d_trained_detectors.md` 为准，不在 README 内重复；BSD-3-Clause 实现许可不能代替数据与权重分发许可。
3. **换真实规程文档**：把演示占位文档换成 GB/DL/T 系列与部委指南，逐份登记版本与日期（`config/sources.yaml` 已有字段）。注意标准原文有版权与获取门槛，需要学校或电网侧的账号。
4. **参赛材料**（**已完成主体**）：`out/低空之眼-参赛方案书.pdf`（9 章 + 第五章附一、附二、附三三篇附章，实测 72 页；由 `tools/build_proposal.py` 从 `docs/proposal/*.md` 汇编，`03_market_size.md` 不存在时 TAM / SAM / SOM 并入第二章）与 `out/demo_screencast.mp4`（32 秒 / 9 个镜头，由 `tools/record_demo.py` 用 CDP 驱动真实浏览器录制）。仍缺的是**团队信息**：`docs/proposal/08_team.md` 里有 24 处 `【待填】` 占位（姓名 / 学校 / 专业年级 / 每周可投入小时 / 是否有指导教师等），填完重跑一次 `tools/build_proposal.py` 即可。

已完成（本轮）：场景文档范围过滤、条件分支校验、参考条款角色分离（依据 / 参考）、窄屏适配、BGE 后端接入与两处拒答缺陷修复，以及配套的回归测试与评测对照。

---

## 十二、合规与来源声明

- 演示图由 `tools/make_demo_assets.py` 用 Pillow **程序化合成**，无第三方图像版权问题；标注为合成标注，已在图片旁与脚本头部声明。
- 演示知识文档为**自编占位文本**，不代表任何标准原文；`config/sources.yaml` 中每份文档标注 `source_version: demo-v1.0` 与「项目自编占位文本」。
- 图标来自 [Lucide](https://lucide.dev)（ISC 许可），本地 sprite 化，运行时不访问外网。
- 检索与评测算法为自行实现（BM25 使用 `rank_bm25`，RRF、词法重排、术语表折叠、指标计算均为本项目代码）。
- 生成后端默认不调用任何外部大模型；启用 `openai_compat` 时数据会发往所配置的服务端，需自行评估数据合规。
- 扩展检测权重（`models/detector/` 下的 FCOS `best` / `last` 与 ONNX）由 `torchvision` 的 `FCOS_ResNet50_FPN_Weights.COCO_V1` COCO 预训练权重在 D-Fire 与 IP102 上微调得到（初始化权重名见 `tools/gpu/train_inspection_fcos.py:239`），`torchvision` 的 BSD-3-Clause 是实现许可、不覆盖数据与权重的分发权。**D-Fire 与 IP102 的原图不随本仓库分发**；由两者微调得到的权重仅用于本地评测与演示，对外分发前需自行取得数据方授权（IP102：学术用途免费、其他用途须联系作者，竞赛 / 商业与再分发许可尚未确立）。逐项许可记录见 `out/inspection_gpu/dfire_source_license.json` 与 `out/inspection_gpu/ip102_source_license.json`，详见 7.7 节。
