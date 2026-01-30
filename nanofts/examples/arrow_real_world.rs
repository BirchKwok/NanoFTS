//! Real-world Arrow integration benchmark
//! 
//! This benchmark simulates the realistic scenario where data arrives
//! in Arrow format and needs to be imported into NanoFTS.
//! 
//! It compares:
//! 1. Arrow → Vec<String> → add_documents_columnar (current approach)
//! 2. Arrow → &str slices → add_documents_arrow_str (zero-copy)

use nanofts::{UnifiedEngine, EngineConfig};

/// Simulate an Arrow StringArray buffer
/// In real Arrow, all strings are stored in a single contiguous buffer
struct SimulatedArrowArray {
    buffer: String,       // All strings concatenated
    offsets: Vec<usize>,  // Offsets into buffer for each string
}

impl SimulatedArrowArray {
    fn new(strings: &[String]) -> Self {
        let mut buffer = String::with_capacity(strings.iter().map(|s| s.len()).sum());
        let mut offsets = vec![0];
        
        for s in strings {
            buffer.push_str(s);
            offsets.push(buffer.len());
        }
        
        Self { buffer, offsets }
    }
    
    /// Get string slice at index (zero-copy view into buffer)
    fn value(&self, idx: usize) -> &str {
        &self.buffer[self.offsets[idx]..self.offsets[idx + 1]]
    }
    
    /// Convert to Vec<String> (requires allocation)
    fn to_vec_string(&self) -> Vec<String> {
        (0..self.offsets.len() - 1)
            .map(|i| self.value(i).to_string())
            .collect()
    }
    
    /// Get all values as string slices (zero-copy)
    fn to_slices(&self) -> Vec<&str> {
        (0..self.offsets.len() - 1)
            .map(|i| self.value(i))
            .collect()
    }
    
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Real-World Arrow Integration Benchmark                     ║");
    println!("║     Scenario: Arrow StringArray → NanoFTS                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    let sizes = [1_000, 10_000, 50_000, 100_000];
    
    // Generate test data
    println!("Generating test data and simulating Arrow arrays...");
    
    let all_titles: Vec<Vec<String>> = sizes.iter()
        .map(|&size| {
            (0..size)
                .map(|i| format!("Document Title {} with keywords", i))
                .collect()
        })
        .collect();
    
    let all_contents: Vec<Vec<String>> = sizes.iter()
        .map(|&size| {
            (0..size)
                .map(|i| format!("Content of document {} for search testing", i))
                .collect()
        })
        .collect();
    
    let all_doc_ids: Vec<Vec<u32>> = sizes.iter()
        .map(|&size| (1..=size as u32).collect())
        .collect();
    
    // Simulate Arrow arrays
    let all_arrow_titles: Vec<SimulatedArrowArray> = all_titles.iter()
        .map(|titles| SimulatedArrowArray::new(titles))
        .collect();
    
    let all_arrow_contents: Vec<SimulatedArrowArray> = all_contents.iter()
        .map(|contents| SimulatedArrowArray::new(contents))
        .collect();
    
    println!("Test data ready.\n");
    
    for (idx, &size) in sizes.iter().enumerate() {
        let doc_ids = &all_doc_ids[idx];
        let arrow_titles = &all_arrow_titles[idx];
        let arrow_contents = &all_arrow_contents[idx];
        
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Dataset Size: {} documents                                    │", size);
        println!("└─────────────────────────────────────────────────────────────────┘");
        
        // ═══════════════════════════════════════════════════════════════
        // Approach 1: Convert Arrow to Vec<String> (current approach)
        // This is what Python from_arrow() does internally
        // ═══════════════════════════════════════════════════════════════
        let (time_convert, time_import) = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            // Step 1: Convert Arrow StringArray to Vec<String>
            let start = std::time::Instant::now();
            let titles_owned = arrow_titles.to_vec_string();
            let contents_owned = arrow_contents.to_vec_string();
            let convert_time = start.elapsed();
            
            // Step 2: Import into NanoFTS
            let start = std::time::Instant::now();
            let columns = vec![
                ("title".to_string(), titles_owned),
                ("content".to_string(), contents_owned),
            ];
            engine.add_documents_columnar(doc_ids.clone(), columns).unwrap();
            let import_time = start.elapsed();
            
            (convert_time, import_time)
        };
        
        println!(
            "  Arrow→Vec<String>→columnar:  convert={:>6.2?} + import={:>6.2?} = {:>6.2?}",
            time_convert,
            time_import,
            time_convert + time_import
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Approach 2: Zero-copy Arrow to &str slices
        // ═══════════════════════════════════════════════════════════════
        let (time_zero_copy, _) = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            let start = std::time::Instant::now();
            
            // Zero-copy: directly use string slices from Arrow buffer
            let title_slices = arrow_titles.to_slices();
            let content_slices = arrow_contents.to_slices();
            
            let columns_slices = vec![
                ("title".to_string(), title_slices),
                ("content".to_string(), content_slices),
            ];
            
            engine.add_documents_arrow_str(doc_ids, columns_slices).unwrap();
            let elapsed = start.elapsed();
            
            (elapsed, elapsed)
        };
        
        println!(
            "  Arrow→&str→arrow_str (0-copy):{:>33.2?}",
            time_zero_copy
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Approach 3: Pre-merge then zero-copy
        // ═══════════════════════════════════════════════════════════════
        let time_pre_merge = {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            let start = std::time::Instant::now();
            
            // Pre-merge strings (still zero-copy from Arrow buffers)
            let merged: Vec<String> = (0..arrow_titles.len())
                .map(|i| format!("{} {}", arrow_titles.value(i), arrow_contents.value(i)))
                .collect();
            
            engine.add_documents_texts(doc_ids.clone(), merged).unwrap();
            start.elapsed()
        };
        
        println!(
            "  Arrow→pre-merge→texts:       {:>33.2?}",
            time_pre_merge
        );
        
        // ═══════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════
        println!("\n  ─────────────────────────────────────────────────────────────");
        let total_convert = time_convert + time_import;
        
        let speedup_zero_copy = total_convert.as_secs_f64() / time_zero_copy.as_secs_f64();
        let speedup_pre_merge = total_convert.as_secs_f64() / time_pre_merge.as_secs_f64();
        
        println!("  Zero-copy speedup:   {:.2}x {}", 
            speedup_zero_copy,
            if speedup_zero_copy > 1.0 { "🚀" } else { "" }
        );
        println!("  Pre-merge speedup:   {:.2}x {}", 
            speedup_pre_merge,
            if speedup_pre_merge > 1.0 { "🚀" } else { "" }
        );
        
        // Calculate overhead of String conversion
        let convert_overhead = time_convert.as_secs_f64() / total_convert.as_secs_f64() * 100.0;
        println!("  Conversion overhead: {:.1}% of total time", convert_overhead);
        
        println!();
    }
    
    // ═════════════════════════════════════════════════════════════════
    // Memory analysis
    // ═════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Memory Layout Comparison                                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Arrow StringArray Layout (contiguous storage):");
    println!("  ┌────────────────────────────────────────────────────────┐");
    println!("  │  Buffer: [\"Hello\"][\"World\"][\"Foo\"]...              │");
    println!("  │          ↑      ↑       ↑                              │");
    println!("  │  Offsets:[0,    5,      10,     13]                    │");
    println!("  └────────────────────────────────────────────────────────┘");
    println!();
    println!("  Vec<String> Layout (scattered allocation):");
    println!("  ┌────────────────────────────────────────────────────────┐");
    println!("  │  Vec[ptr1, ptr2, ptr3]                                 │");
    println!("  │   ↓      ↓      ↓                                      │");
    println!("  │  [Heap 1] [Heap 2] [Heap 3] ...scattered allocations   │");
    println!("  └────────────────────────────────────────────────────────┘");
    println!();
    
    let sample_size = 100_000;
    let avg_str_len = 50;
    
    let arrow_buffer_size = sample_size * avg_str_len;
    let arrow_offsets_size = sample_size * 8; // usize = 8 bytes
    let arrow_total = arrow_buffer_size + arrow_offsets_size;
    
    let vec_string_overhead = sample_size * 24; // String struct (ptr + len + cap)
    let vec_allocation_overhead = sample_size * 16; // Allocator metadata estimate
    let vec_total = arrow_buffer_size + vec_string_overhead + vec_allocation_overhead;
    
    println!("  For {} strings (~{} bytes each):", sample_size, avg_str_len);
    println!("  ├─ Arrow StringArray:   {:>6} MB", arrow_total / 1024 / 1024);
    println!("  │   ├─ Buffer:          {:>6} MB", arrow_buffer_size / 1024 / 1024);
    println!("  │   └─ Offsets:         {:>6} KB", arrow_offsets_size / 1024);
    println!("  ├─ Vec<String>:         {:>6} MB", vec_total / 1024 / 1024);
    println!("  │   ├─ String data:     {:>6} MB", arrow_buffer_size / 1024 / 1024);
    println!("  │   ├─ String structs:  {:>6} MB", vec_string_overhead / 1024 / 1024);
    println!("  │   └─ Allocator overhead:{:>4} MB", vec_allocation_overhead / 1024 / 1024);
    println!("  └─ Memory savings:      {:>6.1}%", 
        (vec_total - arrow_total) as f64 / vec_total as f64 * 100.0);
    println!();
    
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    Real-World Benchmark Complete                   ");
    println!("═══════════════════════════════════════════════════════════════════");
}
