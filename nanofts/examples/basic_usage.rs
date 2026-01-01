//! Basic usage example for NanoFTS
//!
//! Run with: cargo run --example basic_usage

use nanofts::{UnifiedEngine, EngineConfig, EngineResult};
use std::collections::HashMap;

fn main() -> EngineResult<()> {
    println!("=== NanoFTS Basic Usage Example ===\n");

    // 1. Create an in-memory search engine
    println!("1. Creating in-memory search engine...");
    let engine = UnifiedEngine::new(EngineConfig::memory_only())?;
    println!("   Engine created: {}\n", engine);

    // 2. Add some documents
    println!("2. Adding documents...");
    let documents = vec![
        (1, "Rust programming language", "Rust is a systems programming language focused on safety and performance"),
        (2, "Python programming", "Python is a high-level programming language known for simplicity"),
        (3, "Full-text search engines", "Search engines index documents for fast retrieval"),
        (4, "Database systems", "Databases store and manage structured data efficiently"),
        (5, "Machine learning basics", "Machine learning enables computers to learn from data"),
    ];

    for (id, title, content) in documents {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), title.to_string());
        fields.insert("content".to_string(), content.to_string());
        engine.add_document(id, fields)?;
        println!("   Added document {}: {}", id, title);
    }
    println!();

    // 3. Basic search
    println!("3. Searching for 'programming'...");
    let result = engine.search("programming")?;
    println!("   Found {} documents in {:.3}ms", result.total_hits(), result.elapsed_ms());
    println!("   Document IDs: {:?}\n", result.to_list());

    // 4. Search with multiple terms
    println!("4. Searching for 'rust safety'...");
    let result = engine.search("rust safety")?;
    println!("   Found {} documents in {:.3}ms", result.total_hits(), result.elapsed_ms());
    println!("   Document IDs: {:?}\n", result.to_list());

    // 5. Boolean operations
    println!("5. Boolean search operations...");
    
    // AND search
    let and_result = engine.search_and(vec!["programming".to_string(), "language".to_string()])?;
    println!("   'programming' AND 'language': {} documents", and_result.total_hits());
    
    // OR search
    let or_result = engine.search_or(vec!["rust".to_string(), "python".to_string()])?;
    println!("   'rust' OR 'python': {} documents", or_result.total_hits());
    println!();

    // 6. Result set operations
    println!("6. Result set operations...");
    let result1 = engine.search("programming")?;
    let result2 = engine.search("language")?;
    
    let intersection = result1.intersect(&result2);
    println!("   Intersection of 'programming' and 'language': {} documents", intersection.total_hits());
    
    let union = result1.union(&result2);
    println!("   Union of 'programming' and 'language': {} documents", union.total_hits());
    
    let difference = result1.difference(&result2);
    println!("   Difference (programming - language): {} documents", difference.total_hits());
    println!();

    // 7. Fuzzy search
    println!("7. Fuzzy search for 'programing' (misspelled)...");
    let fuzzy_result = engine.fuzzy_search("programing", 1)?;
    println!("   Found {} documents (fuzzy: {})", fuzzy_result.total_hits(), fuzzy_result.is_fuzzy_used());
    println!("   Document IDs: {:?}\n", fuzzy_result.to_list());

    // 8. Pagination
    println!("8. Pagination...");
    let all_result = engine.search_or(vec![
        "rust".to_string(), 
        "python".to_string(), 
        "search".to_string(),
        "database".to_string(),
        "machine".to_string(),
    ])?;
    println!("   Total results: {}", all_result.total_hits());
    println!("   Page 1 (offset=0, limit=2): {:?}", all_result.page(0, 2));
    println!("   Page 2 (offset=2, limit=2): {:?}", all_result.page(2, 2));
    println!();

    // 9. Statistics
    println!("9. Engine statistics:");
    let stats = engine.stats();
    println!("   Term count: {}", engine.term_count());
    println!("   Search count: {}", stats.get("search_count").unwrap_or(&0.0));
    println!("   Cache hit rate: {:.2}%", stats.get("cache_hit_rate").unwrap_or(&0.0) * 100.0);
    println!();

    // 10. Document operations
    println!("10. Document update and delete...");
    
    // Update document
    let mut updated_fields = HashMap::new();
    updated_fields.insert("title".to_string(), "Rust programming language updated".to_string());
    updated_fields.insert("content".to_string(), "Rust is now even better!".to_string());
    engine.update_document(1, updated_fields)?;
    println!("    Updated document 1");
    
    // Delete document
    engine.remove_document(5)?;
    println!("    Deleted document 5");
    
    // Verify
    let search_ml = engine.search("machine learning")?;
    println!("    Search 'machine learning' after delete: {} documents", search_ml.total_hits());
    println!();

    println!("=== Example completed successfully! ===");
    Ok(())
}

