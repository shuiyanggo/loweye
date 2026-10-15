# 第五章 原型与演示

## 5.0 本章的标注规则

与第一章、第七章、第九章一致：

| 标注 | 含义 |
| --- | --- |
| 【事实】 | 可被第三方复核的客观信息（代码、配置文件、截图、外部公开信息） |
| 【实测】 | 本项目在本机跑出的数字，给出测试条件与出处 |
| 【目标】 | 尚未达成、写进里程碑的指标 |
| 【待核实】 | 来源不够硬或尚未取得原文，不得当作结论使用 |
| 【判断】 | 团队主观观点 |

本章描述的是**已经在仓库里跑起来的原型**，不是设计稿。凡本章出现的界面元素，都能在 `web/index.html`、`web/app.js`、`web/styles.css` 里找到对应实现并给出行号；凡本章出现的接口，都能在 `loweye/api.py` 里找到路由。【事实】

【判断】作品赛的评审会按方案书里画出的每一块去追问实现，因此本章有意不做界面美化式描述：只写「这一屏展示了什么数据、这些数据从哪个字段来、点下去会发生什么」。

---

## 5.1 原型形态与运行方式

### 5.1.1 一句话形态

【事实】Web 控制台是一个**免构建、无第三方运行时依赖、断网可跑**的静态页：`web/` 目录下 3 个文件共 1958 行（`index.html` 241 行、`styles.css`、`app.js` 911 行），加一个本地图标 sprite。没有 npm、没有打包器、没有 CDN、没有前端框架。

【判断】这个选择的理由不是「简单」，而是**演示可靠性**：评审现场的网络、浏览器版本、Node 环境都不受我们控制，而一个需要 `pnpm install` 才能在评委机器上打开的原型，失败方式不可穷举。免构建把这个风险压到只剩「浏览器能不能开一个本地端口」。

### 5.1.2 启动与访问

```powershell
# 演示前 30 秒准备：回到零工单，然后起服务
.venv\Scripts\python.exe tools\reset_demo.py --yes
.venv\Scripts\python.exe -m loweye.cli.serve --host 127.0.0.1 --port 8848 --no-reload
# 浏览器打开 http://127.0.0.1:8848/
```

（出处：README 第三节、第八节；端口取自 `config/app.yaml:103-104`）

【事实】三点工程约定，都写在 README 第九节：

- **改完后端代码必须重启服务**：`--reload` 在本机不可靠（改 `schemas.py` 后接口仍按旧契约校验），所以启动脚本默认 `--no-reload`。
- **`/` 静态挂载必须最后挂**：Starlette 按注册顺序匹配，先挂 `/` 会把 `/assets/*` 全截走（表现为 404）。这条写在 `loweye/api.py:319-334` 的 `mount_static()` 注释里。
- **交互式接口文档**：服务起来后访问 `http://127.0.0.1:8848/docs`。

### 5.1.3 前端与后端的契约

【事实】控制台消费 12 个 API 路由中的 9 个；3 个静态挂载分别是 `/`（前端）、`/assets`（演示图）、`/uploads`（上传件）。

| 前端动作 | 接口 | 控制台读的字段 |
| --- | --- | --- |
| 顶栏状态灯（`#health-pill`） | `GET /api/health` | `status`、`backends`（检测器/生成器/嵌入器/重排器/BM25 后端名）、`warnings` |
| 演示图库（`#asset-grid`） | `GET /api/assets` | `image_id`、`url`、`has_annotation` |
| 点图跑链路 | `POST /api/analyze` | 完整 `PipelineResult`（检测、诊断、工单、`timings`、`degradations`） |
| 拖入或点选上传 | `POST /api/analyze/upload` | 同上 |
| 知识库页检索 | `POST /api/search` | `hits[].scores`（四路）、`covered`、`scope_excluded`、`matched_terms`、`search_query` |
| 知识库页统计（`#kb-stats`） | `GET /api/kb/stats` | 知识库统计 + **`min_relevance`** + 术语表条数 + `degradations` |
| 工单页（`#order-list`） | `GET /api/orders` | `{"items":[...],"count":N}` |
| 确认派单 / 退回 | `POST /api/orders/{id}/review` | `status`、`route`、`reviewer`、`decided_at` |
| 追溯页（`#trace-list`） | `GET /api/traces`、`GET /api/traces/{id}` | `n_detections`、`n_orders`、`n_human_review`、`n_degradations`、`total_ms`、`max_severity` |
| PDF 报告按钮（`#btn-report-pdf`） | `GET /api/report/{trace_id}?fmt=html\|pdf` | 文件流（非法 `fmt` 返回 400） |
| 全链路自检（`#btn-selfcheck`） | `POST /api/selfcheck` | 六步逐步 `ok` 与 `ms` |

【事实】`/api/kb/stats` 有一个设计细节值得单独指出：**它把 `min_relevance` 一起回传**，理由写在 `loweye/api.py:127-128`：「前端要能显示『为什么拒答』，而不是把 0.18 这个数字硬编码在页面里（改配置就对不上了）」。【判断】这是「阈值只有一个出处」这条纪律在前端上的落点——页面上显示的阈值与后端用来判定的阈值不可能不一致，因为它们读的是同一个配置项。

---

## 5.2 四个页签各自展示什么

页签结构在 `web/index.html:38-53`：分析台（`data-view="console"`）、工单（`orders`，带计数徽标 `#orders-badge`）、知识库（`kb`）、追溯（`traces`）。页签下有一条形变滑块 `#tab-ink`。【事实】

### 5.2.1 分析台（`#view-console`）

三栏结构：左栏两栏面板 + 中间舞台列。

| 区域 | 元素 | 展示什么 |
| --- | --- | --- |
| 左栏 | `#asset-grid`（`index.html:68`） | 4 张演示图缩略卡；点卡片即跑链路（`app.js:226`） |
| 左栏 | 上传框 `#upload-drop` / `#file-input`（`:70-75`） | 拖入或点选图片，走 `POST /api/analyze/upload`；文案「上传后走同一条链路 结果不落公网」 |
| 左栏 | 运行后端 `#backend-list` / `#backend-note`（`:78-82`） | 当前生效的检测/生成/嵌入/重排/BM25 后端名与降级说明 |
| 舞台 | 图片 `#stage-img` + 叠加层 `#overlay`（`:100-102`） | 检测框按下发严重度着色：`el.className = 'det-box sev-' + sev`（`app.js:364`）；每框带类别标签与置信度；框的进出都有动画（`clearWithExit`，`app.js:263`、`:344`） |
| 舞台 | 场景 chip `#chip-scene`、追溯号 chip `#chip-trace`（`:91-92`） | 本次分析命中的场景（`power` / `agriculture` / `emergency`）与追溯号 |
| 舞台 | PDF 报告按钮 `#btn-report-pdf`（`:93-95`） | 分析完成前 `disabled`，完成后可下载 |
| 舞台 | 流水线 `#pipeline`（`:111-129`） | 四个节点 `detect` / `diagnose` / `grade_dispatch` / `archive`，按**真实耗时**依次点亮；进度轨 `#ptrack-fill` 宽度随节点推进（`app.js:235-253`） |
| 舞台 | `#stage-meta`（`:131`） | 本次分析的分阶段耗时与降级项 |
| 中列 | 诊断卡 `#findings`（`:134-140`） | 每个缺陷一张卡：类别、置信度、等级、责任岗位、时限、动作句；每条动作带条款 chip（`cite-chip`，`app.js:453`），点击弹出条款原文（`openClause`，`app.js:470`） |
| 中列 | 依据条款 / 参考条款分离（`app.js:441-446`） | `role === 'basis'` 归入「依据条款」，其余归入「参考条款」（带 `is-ref` 样式） |
| 底 | 工单条 `#orders-strip`（`:142`） | 本次分析产生的工单摘要 |

【判断】分析台里最能说明问题的是**「依据条款」与「参考条款」并排**这个小设计。它让一次分析里「命中了但没被采用」的条款也留在屏幕上：火情工单的依据是 3.1.1／3.2.3，而「一级／二级响应」判据以参考条款出现。人因此能核对「P1 是怎么判出来的」，而不只是接受一个等级。【事实】

### 5.2.2 工单（`#view-orders`）

| 元素 | 展示什么 |
| --- | --- |
| 分段筛选器 `#order-filter`（`index.html:151-156`） | 全部 / 待人工确认 / P1 / P2 四个筛选 |
| 工单列表 `#order-list`（`:158`） | 工单卡：工单号、等级、责任岗位、SLA 时限、状态、`route`、动作清单、依据条款 |
| 操作 | 「确认派单」「退回」按钮触发 `reviewOrder()`（`app.js:588`），调 `POST /api/orders/{id}/review` |
| 顶栏徽标 `#orders-badge` | 工单数；新增时有一次弹跳动画（`.tab-badge.is-bump`，`styles.css:256`） |

【事实】复核的三种决策在服务端映射为（`loweye/api.py:236-245`）：

| 决策 | 结果 |
| --- | --- |
| `approve` | `status=dispatched`、`route=auto` |
| `reject` | `status=closed` |
| `amend` | `status=dispatched`、`route=auto`，且用 `amended_actions` 覆盖动作清单 |

【判断】`amend` 这一路是刻意留的：复核人不只能「同意/不同意」，还能改掉动作再派。理由与第十章一致——系统的使用者是复核者，不是被指挥的人；如果只能全盘接受或全盘退回，人就会绕过系统直接干活。

### 5.2.3 知识库（`#view-kb`）

| 元素 | 展示什么 |
| --- | --- |
| 检索框 `#search-form` / `#search-input`（`index.html:170-175`） | 自由文本检索，`placeholder` 给两个例子 |
| 提示按钮 `.hint`（`:176-182`） | 四个预置问题，覆盖四种结果：正常命中、「绝缘子自爆怎么处理」；口语改写、「瓷瓶上落了厚厚一层灰 算几级」；知识库边界、「红外测温的温升判据是多少」；越界、「今天股市行情怎么样」 |
| 命中列表 `#hit-list`（`:183`） | 每条命中的条款号、文档名、原文片段，以及**四路分数条** |
| 检索链路图例 `.stage-legend`（`:192-197`） | BM25（关键词字面命中）/ 向量（语义相似召回）/ RRF（两路排名融合）/ 重排（决定最终排序与是否覆盖） |
| 统计 `#kb-stats`（`:198`） | 知识库统计、覆盖阈值、术语表条数、降级项 |

【实测】四路分数条不是装饰：每条命中渲染四条 `.sbar`（`sbar-bm25` / `sbar-vec` / `sbar-rrf` / `sbar-re`，`app.js:706-709`），宽度按各路的数值取百分比（`app.js:716-718`）。【事实】

【事实】判定文案有三种（`app.js:722`），逐字为：

| 文案 | 触发条件 |
| --- | --- |
| `检索命中` | `covered = true` |
| `超出覆盖范围 拒答` | `scope_excluded = true`（最近条款本身是范围排除声明） |
| `按设计拒答` | 既未覆盖、也不属范围排除（最高分低于 `min_relevance`） |

【判断】README 第七节 7.2 把拒答写成两类（范围拒答 / 分数不够），实现上是三个文案标签。差异不在语义而在呈现：分数不够那一类又按「命中了但分低」与「几乎没命中」分了两种描述。方案书按实现写三条，避免与界面不一致。

### 5.2.4 追溯（`#view-traces`）

| 元素 | 展示什么 |
| --- | --- |
| 追溯列表 `#trace-list`（`index.html:212`） | 每次分析一行：追溯号、图片、场景、时间、检出数、工单数、待人工确认数、**降级条数**、总耗时、最高等级 |
| 刷新 `#btn-refresh-traces`（`:208-210`） | 重新拉 `GET /api/traces` |

【事实】追溯字段来自 `loweye/api.py:184-209`：`trace_id`、`image_id`、`image_url`、`scene`、`created_at`、`n_detections`、`n_orders`、`n_human_review`、`n_degradations`、`total_ms`、`max_severity`。追溯号形如 `T20261013085052-72786f`（`T` + 时间戳 + 6 位十六进制）。

【判断】把 `n_degradations` 放进**列表页**而不是详情页，是一个有意的选择：降级条数为 0 的分析与降级条数为 3 的分析，可信度不是一回事，而这个信息不该等人点进去才发现。

---

## 5.3 一次分析的完整链路与产物

### 5.3.1 链路

【事实】编排是显式状态机，阶段键固定为四个：`detect`、`diagnose`、`grade_dispatch`、`archive`（`loweye/agent/pipeline.py:160`、`:262`、`:301`、`:316`），业务上对应「识别 → 诊断 → 分级 → 派单 → 归档」五步（分级与派单在同一阶段内完成）：

| 阶段 | 做什么 | 关键参数 | 产物 |
| --- | --- | --- | --- |
| `detect` | 加载图片 → SAHI 切片 → 检测（当前为 sidecar 回放）→ 跨切片 NMS 去重 → 记录 `slice_count` | `detect.conf_threshold: 0.35`、`detect.nms_iou: 0.50`、`detect.slice.size: 640`、`detect.slice.overlap: 0.2` | `Detection` 列表 |
| `diagnose` | 逐检测：类别 → 查询词 → 术语表归一化 → 两路召回 → RRF → 重排 → 三道闸门 → 生成动作句 | `bm25_top_k: 30`、`vector_top_k: 30`、`rrf_k: 60`、`fused_top_k: 20`、`final_top_k: 5`、`min_relevance: 0.18` | `Diagnosis` 列表（依据条款、动作句、参考条款、四路分数） |
| `grade_dispatch` | 查 `config/routes.yaml` 得等级 / 岗位 / 时限 → 判是否强制人工复核 → 建工单 → 落库 → 通知（可选） | `detect.human_review_threshold: 0.60`、`on_severity_in: [P1]` | `WorkOrder` 列表 |
| `archive` | 把整份 `PipelineResult` 写入归档目录并登记工具调用记录 | `app.trace_retention: 200`（内存中保留的最近记录条数） | 归档快照 |

### 5.3.2 三类落盘产物

| 产物 | 路径 | 内容 | 生成方式 |
| --- | --- | --- | --- |
| 工单流 | `out/work_orders.jsonl` | 一行一张工单，可追加、可被外部系统按行消费 | `ToolBox.create_work_order()`（`loweye/agent/tools.py:63`） |
| 报告 | `out/report_{trace_id}.html`、`out/report_{trace_id}.pdf` | 同一份 Jinja2 模板（`config/report_template.html`）渲染，HTML 直接落盘，PDF 由 PyMuPDF Story 排版 | `render_report()`（`loweye/agent/tools.py:168`） |
| 追溯快照 | `out/traces/{trace_id}.json` | 整份 `PipelineResult` 的 JSON（缩进 2），含 `timings`、`degradations`、`work_orders`、`diagnoses` | `ToolBox.archive()`（`loweye/agent/tools.py:158`） |

【事实】报告分三节，与界面口径一致：**降级项 / 分阶段耗时 / 依据条款与参考条款**（README 第八节第 6 步）。

【事实】PDF 这条路上有两处 Windows 相关的工程细节，都写进了 README 第九节：

- **`with_suffix(".tmp.pdf")` 会抛 `ValueError`**（不接受多点后缀），要用 `with_name(stem + ".tmp.pdf")`；且对 `DocumentWriter` 刚写完的文件做原地替换在 Windows 上会 `PermissionError`，必须先写临时名再落最终名。
- **PDF 体积**：直出 3.6 MB（中文回退字体整包嵌入），用 `save(garbage=4, deflate=True, clean=True)` 压到 1.7 MB。

【事实】压缩失败不影响可用性：代码直接把未压缩的原始产物改名交付，并在返回的 `detail` 里注明「未压缩（异常类型）」（`loweye/agent/tools.py:216-231`）。PDF 整体生成失败时**回退输出 HTML**，`ToolResult.ok` 仍为 `True` 但 `degraded=True`，文案写明失败原因与回退文件（`tools.py:233-238`）。【判断】这是一个有意的取舍：报告是交付物，宁可给一份降级的 HTML，也不要给用户一个 500。

【实测】当前仓库里 `out/work_orders.jsonl` 有 4 行工单，`out/traces/` 有归档快照，报告文件按追溯号命名。

---

## 5.4 三分钟演示脚本

【事实】演示前置动作（约 30 秒）：`tools\reset_demo.py --yes` 复位到零工单，再起服务。演示素材是 4 张程序化合成图：`demo_tower_01`、`demo_channel_01`、`demo_field_01`、`demo_fire_01`，每张配同名 `.detections.json` sidecar。

| 时间 | 动作 | 预期屏幕结果 | 讲解要点（口径） |
| --- | --- | --- | --- |
| 0:00–0:15 | 打开 `http://127.0.0.1:8848/`，指顶栏状态灯 | 状态灯为「已连接」，运行后端面板列出当前生效后端名 | 明确说出当前是降级运行：检测是 sidecar（无权重）、向量是 hashing_tfidf、重排是词法。**先自己说，不等评委问** |
| 0:15–0:45 | 分析台点 `demo_tower_01` | 检测框出现：红框 P1 绝缘子自爆 0.93、蓝框 P2 金具锈蚀 0.81；四个节点按真实耗时依次点亮；诊断卡给出 4 条动作，每条带条款号 | 框的颜色按等级着色，不是随机配色；四个节点的时间是实测耗时，不是动画时长 |
| 0:45–1:05 | 点条款 chip，弹出条款原文 | 弹层显示条款号与原文，逐字对得上；下方另有「参考条款」区（如判定句 2.1.1） | 引用是逐字回查的，可核验率 100%（第六章）；「参考条款」是命中但没成为动作的条款，保留下来让人核对判断过程 |
| 1:05–1:30 | 在金具锈蚀卡上指出**没有**「立即申请停电更换」 | 该卡的动作清单里确实不含这条停电建议 | 这是本场最重要的一句：那句只适用于「出现裂纹、变形或磨损」，条件分支闸门把它挡掉了。**少答一条比多答一条更能说明工单能不能用** |
| 1:30–1:50 | 切工单页，P1 工单显示「待人工确认」，点「确认派单」 | 工单状态由待确认变为已派单，`route` 变为 `auto`，复核人与时间落库 | P1 一律人工确认后才派单，是规则表里的条件（`on_severity_in: [P1]`），不是界面限制 |
| 1:50–2:20 | 切知识库页，搜「红外测温的温升判据是多少」 | 出现**范围拒答卡**：最近条款恰好是排除声明，分数高于阈值但判定为「超出覆盖范围 拒答」 | 这道题是「靠阈值拦不住」的活证据：最高分 0.2629，远高于 0.18 |
| 2:20–2:40 | 再搜「瓷瓶上落了厚厚一层灰 算几级」 | 术语条显示「识别到的领域术语 绝缘子·污秽」，命中 2.2.2；四路分数条摊开 | 术语表把口语折叠到规程术语；四路分数让「为什么排第一」可解释 |
| 2:40–3:00 | 切追溯页，下载刚生成的报告 | 报告含「降级项 / 分阶段耗时 / 依据条款与参考条款」三节 | 整条链路的每一步都可回溯，降级没有被藏起来 |

【判断】这份脚本的设计原则是：**把短板放在自己嘴里说出来，而不是等评委挖出来。** 0:00–0:15 与 1:05–1:30 两段都是主动交代——前者交代后端降级，后者展示系统「拒答一条本来能说的话」。演示的目的是让人相信工单能用，而不是让人相信模型很准。

---

## 5.5 深链

【事实】控制台支持四个深链，用于录屏与把某次结果直接发给别人（`web/app.js:894-908`，实现见 `boot()`）：

| 深链 | 行为 |
| --- | --- |
| `/?image=demo_tower_01` | 切到分析台并自动跑这张演示图；找不到时弹提示「没找到这张演示图」 |
| `/?view=kb&q=绝缘子自爆` | 切到知识库页并把查询填进检索框、立即检索 |
| `/?view=orders` | 直接打开工单页 |
| `/?view=traces` | 直接打开追溯页 |

【事实】实现细节：`view` 参数与 URL hash 等价（`params.get('view') || hash`），只接受 `orders`、`kb`、`traces` 三个值；`image` 参数按 `image_id` 在已加载的图库列表里查（`app.js:802-908`）。【判断】深链的用途不只是录屏方便：它让「某一次分析的结果」变成一个可粘贴的地址，评审或同事打开即见，不需要复述操作路径。

【事实】`?view=kb&q=...` 这条深链在 README 第八节的演示提示按钮里也用到了同样四个查询，两者共用同一个检索入口 `doSearch()`（`app.js:646`）。

---

## 5.6 窄屏适配

【事实】断点是 ≤720px，覆盖块写在 `styles.css` 末尾（`styles.css:1085`），另有一个 ≤1180px 的中等断点（`styles.css:301`）。窄屏下做的事：

| 项 | 处理 |
| --- | --- |
| 顶栏 | 换行排布 |
| 状态文案 | `text-overflow: ellipsis` 省略 |
| 自检按钮 | 收成纯图标，可读名交给 `title` 与 `aria-label`（`index.html:31-34`） |
| 页签 | 横向滚动 |
| 左栏（演示图库 / 运行后端） | 下移到识别舞台之后 |
| 流水线与统计格 | 改两列 |
| 工单卡操作按钮 | 换行右对齐 |
| 提示条（toast） | 改为贴底通栏 |

（出处：README 第十节）

【判断】窄屏这块有一条踩坑记录值得写进方案书，因为它是**方法问题而不是样式问题**：`--window-size` 小于约 500px 时布局不会真的变窄——Windows 上浏览器窗口有最小宽度，`--window-size=390,844` 的 `--screenshot` 只是把左上角 390×844 裁出来，布局仍按约 504px 计算，被切掉的元素看起来「正常」，窄屏溢出被完全掩盖。【事实】

因此本项目改用 `tools/probe_layout.py`：走 CDP `Emulation.setDeviceMetricsOverride` 强制视口，同时输出 `getBoundingClientRect` 与 `Page.captureScreenshot`。**判据是 `documentElement.scrollWidth == innerWidth`（无横向溢出）**，不是「看起来没问题」。【事实】

另有一条同源的经验：**断点覆盖必须写在样式表末尾**。`.pipeline` 等基础规则在 `styles.css` 中部，同优先级下后置声明获胜，`@media (max-width: 720px)` 若写在前面会被静默盖掉，表现为「改了没反应」。【事实】

【事实】截图产物可用于复核窄屏结论：`out/mobile_console.png`、`out/mobile_kb.png`、`out/mobile_orders.png`、`out/mobile_traces.png`；桌面版为 `out/ui_console.png`、`out/ui_orders.png`、`out/ui_traces.png`、`out/ui_cite_roles.png`。

---

## 5.7 离线可跑：三个具体做法

【事实】

1. **本地 Lucide 图标 sprite**：`web/vendor/icons.svg`，43 个图标，本地 sprite 化，**运行时不访问外网**。图标来自 Lucide（ISC 许可），页面里以 `<use href="vendor/icons.svg#i-scan-line">` 形式引用（`index.html:19` 等）。不使用 emoji。
2. **免构建静态页**：`index.html` + `styles.css` + `app.js` 三个文件，无打包、无 CDN、无第三方运行时依赖；由 FastAPI 以 `StaticFiles(directory=web)` 挂在 `/`，`html=True`。
3. **零下载的默认后端**：向量用 `hashing_tfidf`（纯 numpy）、重排用 `lexical`、生成用 `template`、检测用 `sidecar`。四项默认实现都不需要下载权重或访问网络。

【判断】这三个做法合起来的效果是：**把演示环境的最小依赖压缩到「Python 虚拟环境 + 浏览器」**。代价是默认检索后端是词法实现、检测没有权重——这两个代价已经写在第四章与第六章的短板里，不用离线能力去掩盖它们。

【事实】依赖外网的两个工具默认关闭：`tools.notify.enabled: false`（钉钉/飞书 webhook，需要 `LOWEYE_NOTIFY_WEBHOOK`）、`tools.weather.enabled: false`（和风天气，需要 `LOWEYE_WEATHER_KEY`）。它们依赖公网，**不进入离线演示的关键路径**（`config/app.yaml:81-87`）。通知失败时的降级文案是「通知未送达，工单仍已落库」（`loweye/agent/pipeline.py:299`）。【判断】这句话本身就是一条产品判断：通知是附加动作，工单落库才是主链路，两者不能互相拖垮。

---

## 5.8 动效规范与可访问性

【事实】前端为项目自定约束，写在 README 第十节：

- **配色**：白（主）/ 黑 / 蓝（辅），红色只用于 P1 与退回操作。变量定义在 `styles.css:7-40`（`--white`、`--paper`、`--ink`、`--muted`、`--line`、`--blue`、`--red` 及各自的 soft/line 变体），阴影三级 `--shadow-s/m/l`，圆角三级 `--r-s/m/l`。
- **三段时间**：`--t-fast: 120ms`、`--t-base: 220ms`、`--t-slow: 420ms`；三条缓动：`--ease`、`--ease-out`、`--ease-spring`（`styles.css:30-35`）。
- **每个元素都有出现、过渡、消失三段动画**：入场的 `panel-in` / `card-in` / `box-in` / `label-drop` / `meta-in` / `toast-in` / `modal-in`，退出的 `panel-out` / `card-out` / `box-out` / `toast-out` / `modal-out` / `view-out`，过渡的 `--t-*` + `transition:` 组合。
- **视图切换优先走 `document.startViewTransition()` 做跨帧形变**，不支持的浏览器退化为关键帧（`app.js:140`）；`prefers-reduced-motion` 时跳过视图过渡（`app.js:9` 读 `matchMedia`，`styles.css:1072` 抑制全部动效）。
- **微交互**：按钮 hover 抬升 + `:active` 缩放、图标 hover 旋转、页签下划线形变滑动（`#tab-ink`，`app.js:874` `moveInk`）、徽标弹跳（`.tab-badge.is-bump`）、置信度条与分数条宽度过渡（`.conf-fill` 720ms、`.sbar-fill` 640ms）、检测框呼吸环（`@keyframes ring`）。

【判断】本文档不把动效当作「好看」来写，因为它在演示里承担了一个具体功能：**流水线节点是按真实耗时依次点亮的**（`app.js:235-253`），动画的推进顺序就是状态机的推进顺序。也就是说，观众看到的那条点亮轨迹，与 `timings` 里的四个阶段是同一件事的两种呈现。

【事实】可访问性方面已做的部分有限但明确：`role="tablist"` / `role="tab"`（`index.html:38-51`）、`aria-hidden` 用于纯装饰元素、状态与提示区用 `role="status"` + `aria-live="polite"`（`:225`）、弹层用 `role="dialog"` + `aria-modal="true"`（`:228`）、图标按钮补 `title` 与 `aria-label`。【待核实】未做屏幕阅读器实测，也没有键盘焦点顺序的完整验证；本章把这两项如实记录为待办，不主张可访问性已达标。

---

## 5.9 演示的诚实边界

【事实】页脚常驻一句话：「检测与知识文档均为演示数据 用于验证链路 不代表模型精度」（`index.html:218-222`）。演示图由 `tools/make_demo_assets.py` 用 Pillow **程序化合成**，检测框读自同名 sidecar。

【判断】这意味着三件事必须在演示与文档里同时守住：

1. **不展示任何检测精度数字。** 评测报告第 6 节明确：mAP=1.0 是自证循环（预测与真值来自同一份 sidecar），只能证明评测脚本可运行，**禁止作为模型精度结论引用**。
2. **检索指标不代表真实规程上的表现。** 知识库是 3 份自编占位文档、68 个条款块，检索指标反映的是检索策略的相对差异。
3. **留出集 Recall@1 是 50%。** 这个数字在演示里也要能说出来：术语表把开发集从 42% 拉到 92%，换一批没见过的口语表达掉到 50%。正确结论是词法路线泛化能力到此为止，要换语义向量 + cross-encoder 重排，**而不是回头改术语表把留出集刷上去**（那等于把留出集变成开发集）。

【待核实】原有三条与本项目无关的历史遗留项在演示中不出现：窄屏 `--window-size` 陷阱（已用 CDP 绕过）、`[hidden]` 被自定义 `display: grid` 盖掉（已加 `[hidden]{display:none!important}`）、headless 截图需加 `--force-prefers-reduced-motion`（否则定格在入场动画中途，整页半透明，极易误判成渲染 bug）。这三条已记录并修复，仅在第九节保留为经验。

---

## 5.10 本章结论

1. **原型形态是免构建静态页 + FastAPI 后端**：前端 3 个文件 1958 行，无 npm、无打包、无 CDN、无第三方运行时依赖，断网可跑。【事实】
2. **四个页签各有明确职责**：分析台（跑链路、看检测框与条款依据）、工单（筛选与人工复核）、知识库（四路分数与两类拒答）、追溯（降级条数与报告下载）。【事实】
3. **一次分析落三类产物**：`out/work_orders.jsonl`（工单流）、`out/report_{trace_id}.html/.pdf`（报告，同一模板两格式）、`out/traces/{trace_id}.json`（完整快照）。【事实】
4. **三分钟演示脚本的重心是主动交代短板**：先说自己处于降级运行，再展示系统「挡掉了一条本来能说的话」。【判断】
5. **深链让结果可分享**：`?image=` / `?view=kb&q=` / `?view=orders` / `?view=traces` 四条，录屏与转发共用同一入口。【事实】
6. **窄屏结论用可验证判据而非观感**：断点 ≤720px，判据是 `documentElement.scrollWidth == innerWidth`，且必须有 CDP 强制视口才可信。【事实】
7. **离线可跑靠三件事**：本地图标 sprite、免构建静态页、零下载的默认后端；需要公网的两个工具默认关闭且不进关键路径。【事实】
8. **动效在演示里承担功能**：流水线节点按真实耗时点亮，动画顺序即状态机顺序；全部动效受 `prefers-reduced-motion` 抑制。【事实】
9. **演示不主张检测精度**：页脚常驻声明，sidecar 自证循环的 mAP 禁止对外引用；留出集 Recall@1 50% 属于必须能当场说出的数字。【事实】
10. **本章列出的每一项都能在代码里定位**：界面元素给 `web/index.html` 行号，交互给 `web/app.js` 行号，样式给 `web/styles.css` 行号，接口给 `loweye/api.py` 行号，产物给 `loweye/agent/tools.py` 行号。【事实】
