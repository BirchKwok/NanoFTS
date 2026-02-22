"""
Benchmark: flush() vs flush_async() + wait_flush()

Measures wall-clock time of:
  1. add_documents_texts  — tokenisation + memory only, no I/O
  2. flush()              — blocking: compress + write + fsync
  3. flush_async()        — returns as soon as engine buffer is drained (disk I/O in bg thread)
  4. wait_flush()         — remaining time for the background thread
  5. search latency immediately after flush_async() (before wait_flush)

Usage:
    python bench_flush.py
"""

import time
import tempfile
import os
import sys

try:
    import nanofts
except ImportError:
    sys.exit(
        "nanofts not installed. Build with:\n"
        "  cd nanofts && maturin develop --release"
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def generate_data(n: int):
    doc_ids = list(range(1, n + 1))
    texts = [
        f"Document {i} title with keywords alpha beta gamma delta epsilon zeta eta theta "
        f"iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
        f"content field number {i} extended description for full-text indexing benchmark"
        for i in range(n)
    ]
    return doc_ids, texts


def _tmp_path(suffix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"nanofts_bench_{suffix}.nfts")


def _cleanup(path: str):
    for p in [path, path + ".wal"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


# ── individual benchmarks ─────────────────────────────────────────────────────

def bench_flush(n: int) -> dict:
    path = _tmp_path("flush")
    _cleanup(path)
    doc_ids, texts = generate_data(n)

    engine = nanofts.UnifiedEngine(index_file=path, drop_if_exists=True)

    t0 = time.perf_counter()
    engine.add_documents_texts(doc_ids, texts)
    ingest_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    terms = engine.flush()
    flush_s = time.perf_counter() - t1

    hits = engine.search("alpha").total_hits

    _cleanup(path)
    return {
        "n": n,
        "ingest_ms": ingest_s * 1000,
        "flush_ms": flush_s * 1000,
        "terms": terms,
        "hits_after_flush": hits,
    }


def bench_flush_async(n: int) -> dict:
    path = _tmp_path("flush_async")
    _cleanup(path)
    doc_ids, texts = generate_data(n)

    engine = nanofts.UnifiedEngine(index_file=path, drop_if_exists=True)

    t0 = time.perf_counter()
    engine.add_documents_texts(doc_ids, texts)
    ingest_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    engine.flush_async()
    async_return_s = time.perf_counter() - t1

    # search is immediately available (data in LsmSingleIndex in-memory buffer)
    hits_immediate = engine.search("alpha").total_hits

    t2 = time.perf_counter()
    terms = engine.wait_flush()
    wait_s = time.perf_counter() - t2
    total_s = time.perf_counter() - t1

    hits_after_wait = engine.search("alpha").total_hits

    _cleanup(path)
    return {
        "n": n,
        "ingest_ms": ingest_s * 1000,
        "async_return_ms": async_return_s * 1000,
        "wait_ms": wait_s * 1000,
        "total_ms": total_s * 1000,
        "terms": terms,
        "hits_immediate": hits_immediate,
        "hits_after_wait": hits_after_wait,
    }


# ── report ────────────────────────────────────────────────────────────────────

def print_flush_row(r: dict):
    print(
        f"  n={r['n']:>7} | ingest {r['ingest_ms']:>8.1f}ms"
        f" | flush() {r['flush_ms']:>8.1f}ms"
        f" | terms={r['terms']}"
        f" | hits={r['hits_after_flush']}"
    )


def print_async_row(r: dict):
    print(
        f"  n={r['n']:>7} | ingest {r['ingest_ms']:>8.1f}ms"
        f" | flush_async() return {r['async_return_ms']:>6.2f}ms"
        f" | wait_flush() {r['wait_ms']:>8.1f}ms"
        f" | total {r['total_ms']:>8.1f}ms"
        f" | terms={r['terms']}"
        f" | hits_immediate={r['hits_immediate']}"
        f" | hits_after_wait={r['hits_after_wait']}"
    )


def main():
    sizes = [10_000, 50_000, 100_000]

    print("=== NanoFTS Python flush() vs flush_async() benchmark ===\n")
    print("Using a temp file-backed engine (persistent mode) so fsync cost is real.\n")

    print("--- Blocking flush() ---")
    flush_results = [bench_flush(n) for n in sizes]
    for r in flush_results:
        print_flush_row(r)

    print("\n--- Async flush_async() + wait_flush() ---")
    async_results = [bench_flush_async(n) for n in sizes]
    for r in async_results:
        print_async_row(r)

    print("\n--- Summary: caller-visible latency (flush time until return) ---")
    print(f"  {'n':>7}  {'flush()':>12}  {'flush_async()':>14}  {'speedup':>8}")
    for f, a in zip(flush_results, async_results):
        speedup = f["flush_ms"] / a["async_return_ms"] if a["async_return_ms"] > 0 else float("inf")
        print(
            f"  {f['n']:>7}  {f['flush_ms']:>10.1f}ms"
            f"  {a['async_return_ms']:>12.2f}ms"
            f"  {speedup:>7.1f}x"
        )

    print("\nKey metric: flush_async() return time should be near-zero.")
    print("Search hits should be identical immediately after flush_async() and after wait_flush().")


if __name__ == "__main__":
    main()
