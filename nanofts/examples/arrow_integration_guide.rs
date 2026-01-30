//! Arrow Integration Guide for NanoFTS
//! 
//! This example demonstrates how to integrate NanoFTS with Apache Arrow
//! for zero-copy data ingestion.

use nanofts::{UnifiedEngine, EngineConfig};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     NanoFTS Arrow Integration Guide                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // ═════════════════════════════════════════════════════════════════
    // Scenario 1: You have data in Arrow format from another system
    // ═════════════════════════════════════════════════════════════════
    println!("SCENARIO 1: Arrow data from external source\n");
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Example: DataFusion, Polars, or Arrow IPC file                  │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");
    
    let batch_size: usize = 50_000;
    
    // Simulate Arrow data (in real use, this comes from Arrow arrays)
    let id_array: Vec<u32> = (1..=batch_size as u32).collect();
    let title_array: Vec<String> = (0..batch_size)
        .map(|i| format!("Title {} with keywords", i))
        .collect();
    let content_array: Vec<String> = (0..batch_size)
        .map(|i| format!("Content of document {} for search testing", i))
        .collect();
    
    println!("  📦 Simulated Arrow RecordBatch:");
    println!("     └─ {} rows, 3 columns (id, title, content)", batch_size);
    println!();
    
    // ═════════════════════════════════════════════════════════════════
    // APPROACH 1: Zero-Copy (Recommended for Arrow sources)
    // ═════════════════════════════════════════════════════════════════
    {
        println!("  Approach 1: Zero-Copy Ingestion (RECOMMENDED)");
        println!("  ──────────────────────────────────────────────");
        
        let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
        
        let start = std::time::Instant::now();
        
        // Zero-copy: create string slices from existing data
        let title_slices: Vec<&str> = title_array.iter().map(|s| s.as_str()).collect();
        let content_slices: Vec<&str> = content_array.iter().map(|s| s.as_str()).collect();
        
        // Use the zero-copy API
        let columns = vec![
            ("title".to_string(), title_slices),
            ("content".to_string(), content_slices),
        ];
        
        engine.add_documents_arrow_str(&id_array, columns).unwrap();
        
        let elapsed = start.elapsed();
        println!("  ✅ Ingested {} docs in {:?}", batch_size, elapsed);
        println!("  📊 Throughput: {:.0} docs/sec", 
            batch_size as f64 / elapsed.as_secs_f64());
        println!();
    }
    
    // ═════════════════════════════════════════════════════════════════
    // APPROACH 2: Traditional (with String allocation)
    // ═════════════════════════════════════════════════════════════════
    {
        println!("  Approach 2: Traditional (Vec<String> allocation)");
        println!("  ──────────────────────────────────────────────");
        
        let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
        
        let start = std::time::Instant::now();
        
        // Clone strings (what you would need to do without zero-copy)
        let columns = vec![
            ("title".to_string(), title_array.clone()),
            ("content".to_string(), content_array.clone()),
        ];
        
        engine.add_documents_columnar(id_array.clone(), columns).unwrap();
        
        let elapsed = start.elapsed();
        println!("  ✅ Ingested {} docs in {:?}", batch_size, elapsed);
        println!("  📊 Throughput: {:.0} docs/sec", 
            batch_size as f64 / elapsed.as_secs_f64());
        println!();
    }
    
    // ═════════════════════════════════════════════════════════════════
    // APPROACH 3: Fastest Path (Pre-merged text)
    // ═════════════════════════════════════════════════════════════════
    {
        println!("  Approach 3: Pre-Merged Text (FASTEST)");
        println!("  ──────────────────────────────────────────────");
        
        let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
        
        let start = std::time::Instant::now();
        
        // Pre-merge strings (requires allocation but faster for search)
        let merged_texts: Vec<String> = title_array.iter()
            .zip(content_array.iter())
            .map(|(t, c)| format!("{} {}", t, c))
            .collect();
        
        engine.add_documents_texts(id_array.clone(), merged_texts).unwrap();
        
        let elapsed = start.elapsed();
        println!("  ✅ Ingested {} docs in {:?}", batch_size, elapsed);
        println!("  📊 Throughput: {:.0} docs/sec", 
            batch_size as f64 / elapsed.as_secs_f64());
        println!();
    }
    
    // ═════════════════════════════════════════════════════════════════
    // Scenario 2: Arrow IPC File
    // ═════════════════════════════════════════════════════════════════
    println!();
    println!("SCENARIO 2: Reading from Arrow IPC file\n");
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Example: Batch processing Arrow IPC files                       │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");
    
    println!("  Code Example:");
    println!("  ```rust");
    println!("  use arrow::ipc::reader::FileReader;");
    println!("  use std::fs::File;");
    println!("  ");
    println!("  // Open Arrow IPC file");
    println!("  let file = File::open(\"data.arrow\").unwrap();");
    println!("  let reader = FileReader::try_new(file, None).unwrap();");
    println!("  ");
    println!("  for batch in reader {{");
    println!("      let batch = batch.unwrap();");
    println!("      ");
    println!("      // Extract columns as StringArray");
    println!("      let id_array = batch.column(0).as_any()");
    println!("          .downcast_ref::<UInt32Array>().unwrap();");
    println!("      let title_array = batch.column(1).as_any()");
    println!("          .downcast_ref::<StringArray>().unwrap();");
    println!("      ");
    println!("      // Zero-copy: convert StringArray to &str slices");
    println!("      let titles: Vec<&str> = title_array.iter()");
    println!("          .map(|s| s.unwrap_or(\"\"))");
    println!("          .collect();");
    println!("      ");
    println!("      // Import into NanoFTS");
    println!("      engine.add_documents_arrow_texts(&ids, &titles)?;");
    println!("  }}");
    println!("  ```");
    println!();
    
    // ═════════════════════════════════════════════════════════════════
    // Performance Recommendations
    // ═════════════════════════════════════════════════════════════════
    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Performance Recommendations                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");
    
    println!("  1️⃣  For small batches (< 10K docs):");
    println!("     → Use add_documents_arrow_str for zero-copy");
    println!("     → Memory efficiency matters more than throughput");
    println!();
    
    println!("  2️⃣  For large batches (> 50K docs):");
    println!("     → Consider add_documents_texts with pre-merged strings");
    println!("     → Single-pass tokenization is faster");
    println!();
    
    println!("  3️⃣  For streaming ingestion:");
    println!("     → Use Arrow → &str slices to minimize allocations");
    println!("     → Pre-allocate buffers and reuse across batches");
    println!();
    
    println!("  4️⃣  Memory considerations:");
    println!("     → Arrow StringArray: ~5 bytes per string overhead");
    println!("     → Vec<String>: ~24 bytes per string overhead");
    println!("     → Zero-copy saves 35%+ memory for large datasets");
    println!();
    
    // ═════════════════════════════════════════════════════════════════
    // API Summary
    // ═════════════════════════════════════════════════════════════════
    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ New Arrow-Aware APIs in UnifiedEngine                           │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");
    
    println!("  // Zero-copy columnar ingestion");
    println!("  pub fn add_documents_arrow_str<'a>(");
    println!("      &self,");
    println!("      doc_ids: &[u32],");
    println!("      columns: Vec<(String, Vec<&'a str>)>,  // Zero-copy views");
    println!("  ) -> EngineResult<usize>");
    println!();
    
    println!("  // Zero-copy single text column");
    println!("  pub fn add_documents_arrow_texts<'a>(");
    println!("      &self,");
    println!("      doc_ids: &[u32],");
    println!("      texts: &[&'a str],  // Zero-copy views");
    println!("  ) -> EngineResult<usize>");
    println!();
    
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                        Integration Guide Complete                  ");
    println!("═══════════════════════════════════════════════════════════════════");
}
