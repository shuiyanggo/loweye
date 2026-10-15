"""量一下本地 BGE 权重在这台机器上的真实开销。

答辩时「CPU 能不能跑得动」是要被问到的，所以这里不估算、直接测：
加载耗时、编码吞吐、向量维度、以及进程峰值内存。默认只测嵌入模型，
`--both` 连重排模型一起测（两个 2 GB 级模型同时常驻最容易触发内存不足）。

用法：
    python -m tools.probe_embedder
    python -m tools.probe_embedder --both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DEFAULTS = {
    "m3": PROJECT / "models" / "bge-m3",
    "reranker": PROJECT / "models" / "bge-reranker-v2-m3",
}


def rss_mb() -> float:
    """当前进程常驻内存（Windows 走 psutil 不可用时退化为读性能计数器）。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.WorkingSetSize / 1024 / 1024
    except Exception:
        return -1.0


def clause_texts() -> list[str]:
    """拿真实条款当测试语料，而不是造几句短文本糊弄。"""
    from loweye.config import load_config
    from loweye.parsing.builder import read_clauses_jsonl

    cfg = load_config()
    clauses = read_clauses_jsonl(cfg.path("data_processed") / "clauses.jsonl")
    return [f"{c.clause_no} {c.hierarchy_path}\n{c.text}" for c in clauses]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="连重排模型一起加载")
    ap.add_argument("--repeat", type=int, default=3, help="查询编码重复次数，取平均")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    texts = clause_texts()
    print(f"语料：{len(texts)} 条条款，平均 {sum(len(t) for t in texts) // max(1, len(texts))} 字")
    print(f"加载前进程内存 {rss_mb():.0f} MB")

    t0 = time.perf_counter()
    model = SentenceTransformer(str(DEFAULTS["m3"]))
    load_s = time.perf_counter() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"[嵌入] bge-m3 加载 {load_s:.1f}s，维度 {dim}，加载后内存 {rss_mb():.0f} MB")

    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    build_s = time.perf_counter() - t0
    print(f"[嵌入] 全量编码 {len(texts)} 条：{build_s:.1f}s（{len(texts) / max(build_s, 1e-9):.1f} 条/秒），"
          f"内存 {rss_mb():.0f} MB，形状 {vectors.shape}")

    query = "绝缘子自爆怎么处理"
    times = []
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"[嵌入] 单条查询编码 平均 {sum(times) / len(times):.1f} ms（{len(times)} 次）")

    if args.both:
        from sentence_transformers import CrossEncoder

        t0 = time.perf_counter()
        cross = CrossEncoder(str(DEFAULTS["reranker"]))
        print(f"[重排] bge-reranker-v2-m3 加载 {time.perf_counter() - t0:.1f}s，内存 {rss_mb():.0f} MB")
        pairs = [[query, t] for t in texts[:20]]
        t0 = time.perf_counter()
        scores = cross.predict(pairs)
        one = time.perf_counter() - t0
        print(f"[重排] 20 对打分 {one * 1000:.0f} ms（单对 {one * 1000 / len(pairs):.0f} ms），"
              f"最高分 {max(float(s) for s in scores):.4f}，内存 {rss_mb():.0f} MB")

    print(f"结束前进程内存 {rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
