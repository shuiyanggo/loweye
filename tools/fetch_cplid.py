"""把 CPLID（中国电力线绝缘子数据集）的一个子集拉到项目内，供域内检测评估用。

数据来源：HuggingFace 镜像上的 `introvoyz043/InsulatorDataSet`（即 CPLID：600 张正常 +
248 张缺陷，Pascal VOC XML 标注）。选镜像而不是原站，是因为原站在 IEEE DataPort，
未登录看不到许可字段、下载也要账号；镜像走 hf-mirror，本机可达且快。

**许可提醒**：CPLID 由 Tao et al. 随论文公开，仓库与条目页面都没有明确 LICENSE。
比赛可作为公开数据集引用并写明来源；商用前必须向数据方确认（见
`docs/proposal/_research_detection.md` 第 1 节）。本脚本只在本地拉取用于域内评测，
不随作品再分发。

用法：
    python -m tools.fetch_cplid --limit 60            # 缺 60 + 正常 60
    python -m tools.fetch_cplid --limit 20 --split defective
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEST = PROJECT / "data" / "external" / "cplid"
REPO = "introvoyz043/InsulatorDataSet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="每个子集各取多少张")
    ap.add_argument("--split", choices=("both", "defective", "normal"), default="both")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    # 用镜像的 HTTP API 列文件，不用 HfApi：HfApi 在本机对镜像端点会连接超时
    # （实测 httpx.ConnectTimeout WinError 10060），而 REST 接口是通的。
    import json
    import urllib.request

    url = f"{args.endpoint}/api/datasets/{REPO}"
    # 镜像对没有 User-Agent 的请求直接回 403，必须带上
    req = urllib.request.Request(url, headers={"User-Agent": "loweye-demo/1.0 (+local evaluation)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    files = [s["rfilename"] for s in info.get("siblings", [])]
    print(f"仓库共 {len(files)} 个文件（{REPO}）")

    wanted: list[str] = []
    if args.split in ("both", "defective"):
        imgs = sorted(f for f in files if f.startswith("data/Defective_Insulators/images/"))
        xmls = sorted(f for f in files if f.startswith("data/Defective_Insulators/labels/defect/"))
        wanted += imgs[: args.limit] + [x for x in xmls[: args.limit]]
    if args.split in ("both", "normal"):
        imgs = sorted(f for f in files if f.startswith("data/Normal_Insulators/images/"))
        xmls = sorted(f for f in files if f.startswith("data/Normal_Insulators/labels/"))
        wanted += imgs[: args.limit] + xmls[: args.limit]

    ok = 0
    for rel in wanted:
        target = DEST / rel.replace("data/", "", 1)
        if target.exists() and target.stat().st_size > 0:
            ok += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                REPO, rel, repo_type="dataset",
                local_dir=str(DEST / "_raw"),
            )
            # hf_hub_download 会保留仓库内相对路径，这里搬到扁平结构
            landed = DEST / "_raw" / rel
            if landed.exists():
                landed.replace(target)
            ok += 1
        except Exception as exc:
            print(f"  失败 {rel}：{type(exc).__name__}: {exc}")
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"完成 {ok}/{len(wanted)} 个文件，合计 {total / 1024 / 1024:.1f} MB → {DEST}")
    return 0 if ok == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
