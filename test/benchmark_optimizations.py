"""
Benchmark script to verify optimization improvements.

Compares:
1. Batch import (add_documents_columnar) vs row-by-row (add_document)
2. Search performance with Chinese text (inline is_chinese_char vs regex)
3. Compact performance

Usage:
    python test/benchmark_optimizations.py
"""
import time
import statistics
import tempfile
import os


def bench(fn, label, warmup=1, rounds=5):
    """Run fn `rounds` times and report median/min/max."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"  {label}: median={med*1000:.2f}ms  min={min(times)*1000:.2f}ms  max={max(times)*1000:.2f}ms")
    return med


def benchmark_import():
    """Compare batch vs row-by-row document import."""
    from nanofts import create_engine

    n = 50_000
    doc_ids = list(range(n))
    titles = [f"Document title number {i}" for i in range(n)]
    contents = [f"This is the content of document {i} with some searchable text" for i in range(n)]

    print(f"\n=== Import Benchmark ({n} docs) ===")

    # --- Batch import (optimized path) ---
    def batch_import():
        engine = create_engine("")
        engine.add_documents_columnar(
            doc_ids,
            [("title", titles), ("content", contents)]
        )
    t_batch = bench(batch_import, "Batch (add_documents_columnar)")

    # --- Row-by-row import (old path) ---
    def row_import():
        engine = create_engine("")
        for i in range(n):
            engine.add_document(i, {"title": titles[i], "content": contents[i]})
    t_row = bench(row_import, "Row-by-row (add_document)", rounds=3)

    speedup = t_row / t_batch if t_batch > 0 else float('inf')
    print(f"  >> Batch is {speedup:.1f}x faster than row-by-row")


def benchmark_import_pandas():
    """Compare from_pandas (now batch) vs simulated old row-by-row."""
    try:
        import pandas as pd
    except ImportError:
        print("\n=== Pandas Import Benchmark: SKIPPED (pandas not installed) ===")
        return

    from nanofts import create_engine

    n = 50_000
    df = pd.DataFrame({
        'id': list(range(n)),
        'title': [f"Document title {i}" for i in range(n)],
        'content': [f"Content for document {i} with keywords" for i in range(n)],
    })

    print(f"\n=== Pandas Import Benchmark ({n} docs) ===")

    # --- Current (batch) ---
    def pandas_batch():
        engine = create_engine("")
        doc_ids = df['id'].astype(int).tolist()
        columns = [
            ('title', df['title'].fillna('').astype(str).tolist()),
            ('content', df['content'].fillna('').astype(str).tolist()),
        ]
        engine.add_documents_columnar(doc_ids, columns)
    t_batch = bench(pandas_batch, "from_pandas (batch)")

    # --- Old (row-by-row) ---
    def pandas_row():
        engine = create_engine("")
        for _, row in df.iterrows():
            engine.add_document(int(row['id']), {
                'title': str(row['title']),
                'content': str(row['content']),
            })
    t_row = bench(pandas_row, "from_pandas (old row-by-row)", rounds=2)

    speedup = t_row / t_batch if t_batch > 0 else float('inf')
    print(f"  >> New from_pandas is {speedup:.1f}x faster")


def benchmark_search():
    """Search benchmark including Chinese text."""
    from nanofts import create_engine

    n = 100_000
    engine = create_engine("")
    doc_ids = list(range(n))
    texts = [f"full text search engine document {i}" for i in range(n)]
    # Add some Chinese docs
    chinese_texts = [f"全文搜索引擎文档 {i}" for i in range(n // 2)]
    engine.add_documents_texts(doc_ids[:n], texts)
    engine.add_documents_texts(list(range(n, n + n // 2)), chinese_texts)

    print(f"\n=== Search Benchmark ({n + n//2} docs) ===")

    queries_en = ["search engine", "full text", "document"]
    queries_cn = ["全文搜索", "搜索引擎", "文档"]

    def search_en():
        for q in queries_en:
            for _ in range(100):
                engine.search(q)

    def search_cn():
        for q in queries_cn:
            for _ in range(100):
                engine.search(q)

    bench(search_en, f"English search (3 queries x 100)")
    bench(search_cn, f"Chinese search (3 queries x 100)")


def benchmark_compact():
    """Compact benchmark."""
    from nanofts import create_engine

    print(f"\n=== Compact Benchmark ===")

    def do_compact():
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bench.idx")
        engine = create_engine(path)
        # Add docs in batches
        for batch in range(10):
            start = batch * 1000
            doc_ids = list(range(start, start + 1000))
            texts = [f"batch {batch} document {i} search text" for i in doc_ids]
            engine.add_documents_texts(doc_ids, texts)
            engine.flush()
        # Remove some docs
        for i in range(0, 5000, 3):
            engine.remove_document(i)
        engine.compact()
        # Cleanup
        try:
            os.remove(path)
            wal_path = path + ".wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)
            os.rmdir(tmpdir)
        except:
            pass

    bench(do_compact, "compact (10k docs, 1667 deletions)", rounds=3)


if __name__ == "__main__":
    print("=" * 60)
    print("  NanoFTS Optimization Benchmark")
    print("=" * 60)

    benchmark_import()
    benchmark_import_pandas()
    benchmark_search()
    benchmark_compact()

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60)
