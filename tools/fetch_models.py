"""把嵌入 / 重排权重拉到项目内。

为什么要写这个脚本而不是直接 sentence_transformers 按名字加载：
1) 本机 huggingface.co 不通（实测超时），必须走 hf-mirror.com 镜像；
2) 全局约定不往 C 盘放东西，默认的 ~/.cache/huggingface 不能用，权重固定落在项目
   的 models/ 下，config/app.yaml 里用 *_model_path 指过去；
3) 只拉推理真正要用的文件（跳过 onnx/、示例图、TensorFlow 权重），省一半以上流量。

用法：
    python -m tools.fetch_models              # 两个都拉
    python -m tools.fetch_models --only reranker
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT / "models"

REPOS = {
    "m3": ("BAAI/bge-m3", "bge-m3"),
    "reranker": ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
}

# 只要推理必需的权重与分词器；onnx/、imgs/、assets/、colbert_linear.pt 这些一律跳过。
# 注意 `**/x` 只能匹配子目录里的文件，根目录的 config.json / model.safetensors 必须
# 另写不带 `**/` 的模式——只写 `**/` 那版会「下载成功但一个权重都没有」。
ALLOW = [
    "*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.py", "*.jinja", "*.tiktoken",
    "**/*.json", "**/*.txt", "**/*.model", "**/*.bin", "**/*.safetensors", "**/*.py",
]

# 拉完必须存在的文件，缺一即视为失败（避免「0 个文件也算成功」）
REQUIRED = {
    "m3": ["config.json", "tokenizer.json", "sentencepiece.bpe.model", "pytorch_model.bin"],
    "reranker": ["config.json", "tokenizer.json", "sentencepiece.bpe.model", "model.safetensors"],
}


def human(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch_hf(key: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPOS[key][0],
        local_dir=str(target),
        allow_patterns=ALLOW,
        max_workers=4,
    )


def fetch_ms(key: str, target: Path) -> None:
    """走 ModelScope（魔搭）。

    huggingface.co 直连不通、hf-mirror.com 实测只有 0.74 MB/s（2.17 GB 要等近一小时），
    魔搭是同一批 BAAI 权重的国内官方镜像，速度通常快一个量级。
    """
    from modelscope import snapshot_download

    snapshot_download(
        REPOS[key][0],
        local_dir=str(target),
        ignore_file_pattern=[
            "onnx/*", "imgs/*", "assets/*", "*.pt", "*.onnx", "*.onnx_data",
            "*.h5", "*.msgpack", "*.tflite", "*.ot", "README.md", ".gitattributes",
        ],
    )


def fetch(key: str, source: str) -> bool:
    repo_id, folder = REPOS[key]
    target = MODELS_DIR / folder
    target.mkdir(parents=True, exist_ok=True)

    print(f"[{key}] {repo_id} → {target}（源：{source}）", flush=True)
    started = time.time()
    try:
        (fetch_ms if source == "ms" else fetch_hf)(key, target)
    except Exception as exc:  # 网络/镜像问题要显式暴露，别让调用方以为拉成功了
        print(f"[{key}] 失败：{type(exc).__name__}: {exc}", flush=True)
        return False
    size = dir_size(target)
    missing = [name for name in REQUIRED[key] if not (target / name).exists()]
    if missing:
        print(f"[{key}] 缺关键文件 {missing}（当前只有 {human(size)}），视为失败", flush=True)
        return False
    print(f"[{key}] 完成：{human(size)}，用时 {time.time() - started:.0f}s", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(REPOS), default=None)
    ap.add_argument("--source", choices=("ms", "hf"), default="ms",
                    help="ms=ModelScope（默认，国内快）；hf=huggingface 镜像")
    args = ap.parse_args()

    # 必须在 import huggingface_hub 之前设好，否则会用已经写死的默认端点
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 镜像不支持 xet 传输
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # 全局约定不往 C 盘写东西，魔搭的默认缓存目录也要挪到项目内
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODELS_DIR / ".ms-cache"))

    keys = [args.only] if args.only else sorted(REPOS)
    ok = all(fetch(k, args.source) for k in keys)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
