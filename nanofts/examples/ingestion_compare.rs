//! Performance comparison: Baseline vs Arrow zero-copy ingestion
//! 
//! This benchmark compares:
//! 1. add_documents_columnar - Current API with Vec<String>
//! 2. add_documents_texts - Fastest path with Vec<String>
//! 3. add_documents_arrow_str - Zero-copy with string slices (simulating Arrow)
//! 4. add_documents_arrow_texts - Zero-copy single column with string slices

use nanofts::{UnifiedEngine, EngineConfig};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     NanoFTS Ingestion: Baseline vs Arrow Zero-Copy             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Test data sizes
    let sizes = [1_000, 10_000, 50_000, 100_000];
    
    // Pre-generate all test data to avoid allocation during timing
    println!("Generating test data...");
    let all_doc_ids: Vec<Vec<u32>> = sizes.iter()
        .map(|&size| (1..=size as u32).collect())
        .collect();
    
    let all_titles: Vec<Vec<String>> = sizes.iter()
        .map(|&size| {
            (0..size)
                .map(|i| format!("Document Title Number {} with searchable keywords", i))
                .collect()
        })
        .collect();
    
    let all_contents: Vec<Vec<String>> = sizes.iter()
        .map(|&size| {
            (0..size)
                .map(|i| format!(
                    "This is the content of document {}. It contains various words for search indexing performance testing. Additional text to simulate realistic document length.", 
                    i
                ))
                .collect()
        })
        .collect();
    
    // Pre-merge texts for fastest path
    let all_merged: Vec<Vec<String>> = all_titles.iter()
        .zip(all_contents.iter())
        .map(|(titles, contents)| {
            titles.iter()
                .zip(contents.iter())
                .map(|(t, c)| format!("{} {}", t, c))
                .collect()
        })
        .collect();
    
    println!("Test data ready. Starting benchmarks...\n");
    
    for (idx, &size) in sizes.iter().enumerate() {
        let doc_ids = &all_doc_ids[idx];
        let titles = &all_titles[idx];
        let contents = &all_contents[idx];
        let merged = &all_merged[idx];
        
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Dataset Size: {} documents                                    │", size);
        println!("└─────────────────────────────────────────────────────────────────┘");
        
        // ═══════════════════════════════════════════════════════════════
        // Test 1: add_documents_columnar (current API)
        // ═══════════════════════════════════════════════════════════════
        let time_columnar = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            let columns = vec![
                ("title".to_string(), titles.clone()),
                ("content".to_string(), contents.clone()),
            ];
            
            let start = std::time::Instant::now();
            engine.add_documents_columnar(doc_ids.clone(), columns).unwrap();
            start.elapsed()
        };
        
        println!(
            "  add_documents_columnar:  {:>10.2?} ({:>10.0} docs/sec)",
            time_columnar,
            size as f64 / time_columnar.as_secs_f64()
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Test 2: add_documents_texts (fastest path with owned Strings)
        // ═══════════════════════════════════════════════════════════════
        let time_texts = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            let start = std::time::Instant::now();
            engine.add_documents_texts(doc_ids.clone(), merged.clone()).unwrap();
            start.elapsed()
        };
        
        println!(
            "  add_documents_texts:     {:>10.2?} ({:>10.0} docs/sec)",
            time_texts,
            size as f64 / time_texts.as_secs_f64()
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Test 3: add_documents_arrow_str (zero-copy columnar)
        // Simulates Arrow StringArray where strings are contiguous in memory
        // ═══════════════════════════════════════════════════════════════
        let time_arrow_str = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            // Create string slices (simulating Arrow buffer views)
            // In real Arrow usage, these would be views into Arrow's buffer
            let title_slices: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();
            let content_slices: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
            
            let columns_slices = vec![
                ("title".to_string(), title_slices),
                ("content".to_string(), content_slices),
            ];
            
            let start = std::time::Instant::now();
            engine.add_documents_arrow_str(doc_ids, columns_slices).unwrap();
            start.elapsed()
        };
        
        println!(
            "  add_documents_arrow_str: {:>10.2?} ({:>10.0} docs/sec)",
            time_arrow_str,
            size as f64 / time_arrow_str.as_secs_f64()
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Test 4: add_documents_arrow_texts (zero-copy single column)
        // Fastest Arrow path with pre-merged text
        // ═══════════════════════════════════════════════════════════════
        let time_arrow_texts = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            // String slices from pre-merged texts
            let text_slices: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
            
            let start = std::time::Instant::now();
            engine.add_documents_arrow_texts(doc_ids, &text_slices).unwrap();
            start.elapsed()
        };
        
        println!(
            "  add_documents_arrow_texts:{:>10.2?} ({:>10.0} docs/sec)",
            time_arrow_texts,
            size as f64 / time_arrow_texts.as_secs_f64()
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Summary & Speedup Analysis
        // ═══════════════════════════════════════════════════════════════
        println!("\n  ─────────────────────────────────────────────────────────────");
        println!("  Speedup Analysis (relative to add_documents_columnar):");
        
        let speedup_texts = time_columnar.as_secs_f64() / time_texts.as_secs_f64();
        let speedup_arrow_str = time_columnar.as_secs_f64() / time_arrow_str.as_secs_f64();
        let speedup_arrow_texts = time_columnar.as_secs_f64() / time_arrow_texts.as_secs_f64();
        
        println!("    add_documents_texts:     {:.2}x {}", 
            speedup_texts,
            if speedup_texts > 1.0 { "🚀" } else { "" }
        );
        println!("    add_documents_arrow_str: {:.2}x {}", 
            speedup_arrow_str,
            if speedup_arrow_str > 1.0 { "🚀" } else { "" }
        );
        println!("    add_documents_arrow_texts:{:.2}x {}", 
            speedup_arrow_texts,
            if speedup_arrow_texts > 1.0 { "🚀" } else { "" }
        );
        
        // Calculate overhead of String allocation vs zero-copy
        let overhead_vs_arrow = (time_columnar.as_secs_f64() - time_arrow_str.as_secs_f64()) 
            / time_columnar.as_secs_f64() * 100.0;
        println!("\n  String allocation overhead: {:.1}%", overhead_vs_arrow);
        
        println!();
    }
    
    // ═════════════════════════════════════════════════════════════════
    // Detailed memory analysis simulation
    // ═════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Memory Allocation Analysis                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    
    let sample_size = 100_000;
    let avg_title_len = 50;
    let avg_content_len = 150;
    
    let total_string_bytes = sample_size * (avg_title_len + avg_content_len);
    let string_metadata_overhead = sample_size * 2 * 24; // String struct overhead (ptr + len + cap)
    let vec_overhead = sample_size * 2 * 24; // Vec overhead
    
    println!("  For {} documents:", sample_size);
    println!("  ├─ Raw text data:          {:>10} MB", total_string_bytes / 1024 / 1024);
    println!("  ├─ String metadata:        {:>10} MB", string_metadata_overhead / 1024 / 1024);
    println!("  ├─ Vec overhead:           {:>10} MB", vec_overhead / 1024 / 1024);
    println!("  └─ Total with allocation:  {:>10} MB", 
        (total_string_bytes + string_metadata_overhead + vec_overhead) / 1024 / 1024);
    println!();
    println!("  With Arrow zero-copy:");
    println!("  ├─ Single buffer for text: {:>10} MB", total_string_bytes / 1024 / 1024);
    println!("  ├─ Offsets array:          {:>10} KB", (sample_size * 2 * 8) / 1024);
    println!("  └─ String slices (&str):   {:>10} KB", (sample_size * 2 * 16) / 1024);
    println!();
    
    let memory_saved = string_metadata_overhead + vec_overhead;
    println!("  💾 Estimated memory saved: {:.1} MB ({:.1}%)", 
        memory_saved as f64 / 1024.0 / 1024.0,
        memory_saved as f64 / (total_string_bytes + string_metadata_overhead + vec_overhead) as f64 * 100.0
    );
    
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                          Benchmark Complete                        ");
    println!("═══════════════════════════════════════════════════════════════════");
}
