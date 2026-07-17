//! Head-to-head microbenchmark for ingest + search + fuzzy + flush (NanoFTS 0.8+, u64 doc ids).

use nanofts::{EngineConfig, UnifiedEngine};
use std::time::Instant;

fn make_docs(n: usize) -> (Vec<u64>, Vec<String>) {
    let ids: Vec<u64> = (1..=n as u64).collect();
    let texts: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "Document Title Number {i} with some content. \
                 This is the content of document {i}. It contains various words \
                 for search indexing performance testing including machine learning."
            )
        })
        .collect();
    (ids, texts)
}

fn bench_ingest(n: usize, track_doc_terms: bool) -> f64 {
    let (ids, texts) = make_docs(n);
    let config = EngineConfig::memory_only().with_track_doc_terms(track_doc_terms);
    let engine = UnifiedEngine::new(config).unwrap();
    let start = Instant::now();
    engine.add_documents_texts(ids, texts).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    n as f64 / elapsed
}

fn bench_search(n: usize, queries: &[&str]) -> (f64, f64, u64) {
    let (ids, texts) = make_docs(n);
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    engine.add_documents_texts(ids, texts).unwrap();

    // --- uncached: disable result cache ---
    engine.set_result_cache_enabled(false);
    for _ in 0..3 {
        for q in queries {
            let _ = engine.search(q).unwrap();
        }
    }
    let start = Instant::now();
    let rounds = 50;
    let mut hits = 0u64;
    for _ in 0..rounds {
        for q in queries {
            hits += engine.search(q).unwrap().total_hits();
        }
    }
    let uncached_qps = (rounds * queries.len()) as f64 / start.elapsed().as_secs_f64();

    // --- cached ---
    engine.set_result_cache_enabled(true);
    for q in queries {
        let _ = engine.search(q).unwrap();
    }
    let start = Instant::now();
    for _ in 0..rounds {
        for q in queries {
            let _ = engine.search(q).unwrap();
        }
    }
    let cached_qps = (rounds * queries.len()) as f64 / start.elapsed().as_secs_f64();

    (uncached_qps, cached_qps, hits)
}

fn bench_fuzzy(n: usize) -> f64 {
    let (ids, texts) = make_docs(n);
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    engine.add_documents_texts(ids, texts).unwrap();

    let queries = ["machne", "lerning", "documnt", "performace", "indxing"];
    for _ in 0..2 {
        for q in &queries {
            let _ = engine.fuzzy_search(q, 1).unwrap();
        }
    }

    let start = Instant::now();
    let rounds = 20;
    for _ in 0..rounds {
        for q in &queries {
            let _ = engine.fuzzy_search(q, 1).unwrap();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (rounds * queries.len()) as f64 / elapsed
}

fn bench_flush(n: usize) -> f64 {
    let dir = std::env::temp_dir().join(format!("nanofts_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench.nfts");
    let path_str = path.to_str().unwrap();

    let (ids, texts) = make_docs(n);
    let config = EngineConfig::persistent(path_str);
    let engine = UnifiedEngine::new(config).unwrap();
    engine.add_documents_texts(ids, texts).unwrap();

    let start = Instant::now();
    engine.flush().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_dir_all(&dir);
    n as f64 / elapsed.max(1e-9)
}

fn main() {
    println!("=== NanoFTS Benchmark ===");
    println!("doc_id: u64 (0.8+)");
    println!();

    for &n in &[10_000usize, 50_000] {
        println!("--- N = {n} ---");
        let ingest_default = bench_ingest(n, true);
        println!("ingest (track_doc_terms=true):  {ingest_default:.0} docs/s");
        let ingest_light = bench_ingest(n, false);
        println!("ingest (track_doc_terms=false): {ingest_light:.0} docs/s");

        let queries = [
            "document",
            "machine learning",
            "performance testing",
            "content",
        ];
        let (uncached_qps, cached_qps, hits) = bench_search(n, &queries);
        println!("search QPS uncached: {uncached_qps:.0}  cached: {cached_qps:.0}  hits_sum={hits}");

        let fuzzy_qps = bench_fuzzy(n);
        println!("fuzzy QPS (5 queries x 20): {fuzzy_qps:.0}");

        let flush_rate = bench_flush(n);
        println!("flush throughput: {flush_rate:.0} docs/s (wall for flush call)");
        println!();
    }
}
