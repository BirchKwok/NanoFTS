//! Baseline benchmark for document ingestion
//! 
//! Tests current ingestion methods:
//! - add_documents_columnar
//! - add_documents_texts

use nanofts::{UnifiedEngine, EngineConfig};

fn main() {
    println!("=== NanoFTS Ingestion Baseline Benchmark ===\n");
    
    // Generate test data
    let sizes = [1_000, 10_000, 100_000];
    
    for &size in &sizes {
        println!("\n--- Testing with {} documents ---", size);
        
        // Generate test data
        let doc_ids: Vec<u32> = (1..=size as u32).collect();
        let titles: Vec<String> = (0..size)
            .map(|i| format!("Document Title Number {} with some content", i))
            .collect();
        let contents: Vec<String> = (0..size)
            .map(|i| format!("This is the content of document {}. It contains various words for search indexing performance testing.", i))
            .collect();
        
        // Pre-merge texts for fastest path
        let merged_texts: Vec<String> = titles.iter()
            .zip(contents.iter())
            .map(|(t, c)| format!("{} {}", t, c))
            .collect();
        
        // Test 1: add_documents_columnar (current columnar API)
        {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            let columns = vec![
                ("title".to_string(), titles.clone()),
                ("content".to_string(), contents.clone()),
            ];
            
            let start = std::time::Instant::now();
            engine.add_documents_columnar(doc_ids.clone(), columns).unwrap();
            let elapsed = start.elapsed();
            
            println!(
                "add_documents_columnar: {:?} ({} docs/sec)",
                elapsed,
                size as f64 / elapsed.as_secs_f64()
            );
        }
        
        // Test 2: add_documents_texts (fastest path - pre-merged text)
        {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            let start = std::time::Instant::now();
            engine.add_documents_texts(doc_ids.clone(), merged_texts.clone()).unwrap();
            let elapsed = start.elapsed();
            
            println!(
                "add_documents_texts:    {:?} ({} docs/sec)",
                elapsed,
                size as f64 / elapsed.as_secs_f64()
            );
        }
        
        // Test 3: Simulate Arrow-like access pattern (columnar but with StringArray conversion)
        // This simulates what would happen if we extract strings from Arrow arrays
        {
            let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
            
            // Simulate extracting strings from Arrow StringArray (clone required)
            let titles_extracted: Vec<String> = titles.iter().map(|s| s.to_string()).collect();
            let contents_extracted: Vec<String> = contents.iter().map(|s| s.to_string()).collect();
            
            let columns = vec![
                ("title".to_string(), titles_extracted),
                ("content".to_string(), contents_extracted),
            ];
            
            let start = std::time::Instant::now();
            engine.add_documents_columnar(doc_ids.clone(), columns).unwrap();
            let elapsed = start.elapsed();
            
            println!(
                "arrow-simulated (clone): {:?} ({} docs/sec)",
                elapsed,
                size as f64 / elapsed.as_secs_f64()
            );
        }
    }
    
    println!("\n=== Baseline Complete ===");
}
