//! Benchmark: flush() vs flush_async() + wait_flush()
//!
//! Measures:
//!   1. add_documents_texts time (tokenisation + memory, no I/O)
//!   2. flush() time          (blocking: compress + write + fsync)
//!   3. flush_async() time    (returns as soon as buffer is drained; disk I/O in background)
//!   4. wait_flush() time     (time the background thread still needed after flush_async returned)
//!   5. Search latency immediately after flush_async() (before wait_flush)
//!
//! Run with:
//!   cargo run --example flush_benchmark --release

use nanofts::{EngineConfig, UnifiedEngine};

fn generate_data(n: usize) -> (Vec<u64>, Vec<String>) {
    let doc_ids: Vec<u64> = (1..=n as u64).collect();
    let texts: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "Document {} title with keywords alpha beta gamma delta epsilon zeta eta theta \
                 iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega \
                 content field number {} extended description for full-text indexing benchmark",
                i, i
            )
        })
        .collect();
    (doc_ids, texts)
}

fn bench_flush(label: &str, n: usize, index_path: &str) {
    let (doc_ids, texts) = generate_data(n);

    // ── ingestion ──────────────────────────────────────────────────────────
    let engine = UnifiedEngine::new(
        EngineConfig::persistent(index_path).with_drop_if_exists(true),
    )
    .unwrap();

    let t0 = std::time::Instant::now();
    engine.add_documents_texts(doc_ids.clone(), texts.clone()).unwrap();
    let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── flush() ────────────────────────────────────────────────────────────
    let t1 = std::time::Instant::now();
    let terms_flushed = engine.flush().unwrap();
    let flush_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let search_after_flush = engine.search("alpha").unwrap().total_hits();

    println!(
        "[{label:>12}] n={n:>7} | ingest {ingest_ms:>8.1}ms | flush() {flush_ms:>8.1}ms \
         | terms={terms_flushed} | hits_after_flush={search_after_flush}"
    );

    let _ = std::fs::remove_file(index_path);
    let _ = std::fs::remove_file(format!("{}.wal", index_path));
}

fn bench_flush_async(label: &str, n: usize, index_path: &str) {
    let (doc_ids, texts) = generate_data(n);

    let engine = UnifiedEngine::new(
        EngineConfig::persistent(index_path).with_drop_if_exists(true),
    )
    .unwrap();

    let t0 = std::time::Instant::now();
    engine.add_documents_texts(doc_ids.clone(), texts.clone()).unwrap();
    let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── flush_async() ──────────────────────────────────────────────────────
    let t1 = std::time::Instant::now();
    engine.flush_async().unwrap();
    let async_return_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // search is available immediately (data in LsmSingleIndex buffer)
    let hits_immediate = engine.search("alpha").unwrap().total_hits();

    // ── wait_flush() ───────────────────────────────────────────────────────
    let t2 = std::time::Instant::now();
    let terms_flushed = engine.wait_flush().unwrap();
    let wait_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let total_async_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let hits_after_wait = engine.search("alpha").unwrap().total_hits();

    println!(
        "[{label:>12}] n={n:>7} | ingest {ingest_ms:>8.1}ms \
         | flush_async() {async_return_ms:>6.2}ms (return) \
         | wait_flush() {wait_ms:>8.1}ms | total {total_async_ms:>8.1}ms \
         | terms={terms_flushed} | hits_immediate={hits_immediate} hits_after_wait={hits_after_wait}"
    );

    let _ = std::fs::remove_file(index_path);
    let _ = std::fs::remove_file(format!("{}.wal", index_path));
}

fn main() {
    println!("=== NanoFTS flush() vs flush_async() benchmark ===\n");
    println!("Using a temp file-backed engine (persistent mode) so fsync cost is real.\n");

    let sizes = [10_000usize, 50_000, 100_000];
    let tmp_dir = std::env::temp_dir();

    println!("--- Blocking flush() ---");
    for &n in &sizes {
        let path = tmp_dir.join("nanofts_bench_flush.nfts");
        bench_flush("flush", n, path.to_str().unwrap());
    }

    println!("\n--- Async flush_async() + wait_flush() ---");
    for &n in &sizes {
        let path = tmp_dir.join("nanofts_bench_flush_async.nfts");
        bench_flush_async("flush_async", n, path.to_str().unwrap());
    }

    println!("\n=== Done ===");
    println!("Key metric: flush_async() return time should be ~0ms (or a few ms for buffer drain).");
    println!("The search is available immediately after flush_async() without waiting for fsync.");
}
