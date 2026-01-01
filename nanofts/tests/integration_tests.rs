//! Integration tests for NanoFTS

use nanofts::{UnifiedEngine, EngineConfig, EngineResult, ResultHandle};
use std::collections::HashMap;

/// Helper to create test documents
fn create_doc(title: &str, content: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    fields.insert("title".to_string(), title.to_string());
    fields.insert("content".to_string(), content.to_string());
    fields
}

// ============================================
// Engine Creation Tests
// ============================================

#[test]
fn test_create_memory_engine() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    assert!(engine.is_memory_only());
    assert_eq!(engine.term_count(), 0);
}

#[test]
fn test_create_persistent_engine() {
    let temp_dir = std::env::temp_dir();
    let index_path = temp_dir.join("test_persistent.nfts");
    
    // Clean up if exists
    let _ = std::fs::remove_file(&index_path);
    
    let config = EngineConfig::persistent(index_path.to_str().unwrap())
        .with_drop_if_exists(true);
    
    let engine = UnifiedEngine::new(config).unwrap();
    assert!(!engine.is_memory_only());
    
    // Clean up
    drop(engine);
    let _ = std::fs::remove_file(&index_path);
    let _ = std::fs::remove_file(index_path.with_extension("nfts.wal"));
}

#[test]
fn test_engine_config_builder() {
    let config = EngineConfig::persistent("test.nfts")
        .with_lazy_load(true)
        .with_cache_size(5000)
        .with_fuzzy_threshold(0.8)
        .with_track_doc_terms(true);
    
    assert_eq!(config.index_file, "test.nfts");
    assert!(config.lazy_load);
    assert_eq!(config.cache_size, 5000);
    assert_eq!(config.fuzzy_threshold, 0.8);
    assert!(config.track_doc_terms);
}

// ============================================
// Document Operations Tests
// ============================================

#[test]
fn test_add_single_document() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    let result = engine.add_document(1, create_doc("Hello World", "This is a test"));
    assert!(result.is_ok());
    
    assert!(engine.term_count() > 0);
}

#[test]
fn test_add_multiple_documents() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    let docs = vec![
        (1, create_doc("Rust Programming", "Learn Rust language")),
        (2, create_doc("Python Tutorial", "Python is great")),
        (3, create_doc("Search Engine", "Full-text search")),
    ];
    
    let count = engine.add_documents(docs).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn test_update_document() {
    let config = EngineConfig::memory_only();
    let mut config = config;
    config.track_doc_terms = true;
    
    let engine = UnifiedEngine::new(config).unwrap();
    
    // Add original document
    engine.add_document(1, create_doc("Original Title", "Original content")).unwrap();
    
    // Search for original
    let result = engine.search("original").unwrap();
    assert_eq!(result.total_hits(), 1);
    
    // Update document
    engine.update_document(1, create_doc("Updated Title", "New content")).unwrap();
    
    // Search for updated content
    let result = engine.search("updated").unwrap();
    assert_eq!(result.total_hits(), 1);
    
    // Original content should not be found
    let result = engine.search("original").unwrap();
    assert_eq!(result.total_hits(), 0);
}

#[test]
fn test_remove_document() {
    let config = EngineConfig::memory_only();
    let mut config = config;
    config.track_doc_terms = true;
    
    let engine = UnifiedEngine::new(config).unwrap();
    
    engine.add_document(1, create_doc("Test Doc", "Content here")).unwrap();
    engine.add_document(2, create_doc("Another Doc", "More content")).unwrap();
    
    // Verify document exists
    let result = engine.search("test").unwrap();
    assert_eq!(result.total_hits(), 1);
    
    // Remove document
    engine.remove_document(1).unwrap();
    
    // Should not find removed document
    let result = engine.search("test").unwrap();
    assert_eq!(result.total_hits(), 0);
    
    // Other document still exists
    let result = engine.search("another").unwrap();
    assert_eq!(result.total_hits(), 1);
}

#[test]
fn test_remove_multiple_documents() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    for i in 1..=5 {
        engine.add_document(i, create_doc(&format!("Doc {}", i), "test content")).unwrap();
    }
    
    engine.remove_documents(vec![1, 2, 3]).unwrap();
    
    assert_eq!(engine.deleted_count(), 3);
}

// ============================================
// Search Tests
// ============================================

#[test]
fn test_basic_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust", "Systems programming language")).unwrap();
    engine.add_document(2, create_doc("Python", "High-level programming")).unwrap();
    
    let result = engine.search("rust").unwrap();
    assert_eq!(result.total_hits(), 1);
    assert!(result.contains(1));
    assert!(!result.contains(2));
}

#[test]
fn test_multi_word_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust Programming", "Learn Rust for systems")).unwrap();
    engine.add_document(2, create_doc("Python Programming", "Learn Python for web")).unwrap();
    engine.add_document(3, create_doc("Rust Web", "Use Rust for web development")).unwrap();
    
    // Search for "rust programming" should find doc 1
    let result = engine.search("rust programming").unwrap();
    assert_eq!(result.total_hits(), 1);
    assert!(result.contains(1));
}

#[test]
fn test_search_and() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust Programming", "Systems language")).unwrap();
    engine.add_document(2, create_doc("Python Programming", "Scripting language")).unwrap();
    engine.add_document(3, create_doc("Rust Tutorial", "Learn Rust basics")).unwrap();
    
    let result = engine.search_and(vec!["rust".to_string(), "programming".to_string()]).unwrap();
    assert_eq!(result.total_hits(), 1);
    assert!(result.contains(1));
}

#[test]
fn test_search_or() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust", "Systems")).unwrap();
    engine.add_document(2, create_doc("Python", "Scripting")).unwrap();
    engine.add_document(3, create_doc("Java", "Enterprise")).unwrap();
    
    let result = engine.search_or(vec!["rust".to_string(), "python".to_string()]).unwrap();
    assert_eq!(result.total_hits(), 2);
    assert!(result.contains(1));
    assert!(result.contains(2));
    assert!(!result.contains(3));
}

#[test]
fn test_search_batch() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust", "Systems")).unwrap();
    engine.add_document(2, create_doc("Python", "Scripting")).unwrap();
    
    let results = engine.search_batch(vec![
        "rust".to_string(),
        "python".to_string(),
        "nonexistent".to_string(),
    ]).unwrap();
    
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].total_hits(), 1);
    assert_eq!(results[1].total_hits(), 1);
    assert_eq!(results[2].total_hits(), 0);
}

#[test]
fn test_fuzzy_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Programming", "Learn to program")).unwrap();
    
    // Misspelled query
    let result = engine.fuzzy_search("programing", 1).unwrap();
    assert!(result.total_hits() >= 1 || result.is_fuzzy_used());
}

// ============================================
// Result Handle Tests
// ============================================

#[test]
fn test_result_handle_operations() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    for i in 1..=10 {
        engine.add_document(i, create_doc(&format!("Doc {}", i), "test content")).unwrap();
    }
    
    let result = engine.search("test").unwrap();
    
    // Test total_hits
    assert_eq!(result.total_hits(), 10);
    
    // Test is_empty
    assert!(!result.is_empty());
    
    // Test len
    assert_eq!(result.len(), 10);
    
    // Test to_list
    let list = result.to_list();
    assert_eq!(list.len(), 10);
    
    // Test top
    let top5 = result.top(5);
    assert_eq!(top5.len(), 5);
    
    // Test page
    let page1 = result.page(0, 3);
    let page2 = result.page(3, 3);
    assert_eq!(page1.len(), 3);
    assert_eq!(page2.len(), 3);
    
    // Test contains
    assert!(result.contains(1));
    assert!(!result.contains(100));
    
    // Test iter
    let count = result.iter().count();
    assert_eq!(count, 10);
}

#[test]
fn test_result_set_intersection() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust Programming", "Systems")).unwrap();
    engine.add_document(2, create_doc("Python Programming", "Scripting")).unwrap();
    engine.add_document(3, create_doc("Rust Tutorial", "Learning")).unwrap();
    
    let rust_docs = engine.search("rust").unwrap();
    let programming_docs = engine.search("programming").unwrap();
    
    let intersection = rust_docs.intersect(&programming_docs);
    assert_eq!(intersection.total_hits(), 1);
    assert!(intersection.contains(1));
}

#[test]
fn test_result_set_union() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust", "Systems")).unwrap();
    engine.add_document(2, create_doc("Python", "Scripting")).unwrap();
    
    let rust_docs = engine.search("rust").unwrap();
    let python_docs = engine.search("python").unwrap();
    
    let union = rust_docs.union(&python_docs);
    assert_eq!(union.total_hits(), 2);
}

#[test]
fn test_result_set_difference() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust Programming", "Systems")).unwrap();
    engine.add_document(2, create_doc("Python Programming", "Scripting")).unwrap();
    
    let programming_docs = engine.search("programming").unwrap();
    let python_docs = engine.search("python").unwrap();
    
    let difference = programming_docs.difference(&python_docs);
    assert_eq!(difference.total_hits(), 1);
    assert!(difference.contains(1));
    assert!(!difference.contains(2));
}

#[test]
fn test_filter_by_ids() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    for i in 1..=10 {
        engine.add_document(i, create_doc("Test", "Content")).unwrap();
    }
    
    let result = engine.search("test").unwrap();
    let filtered = engine.filter_by_ids(&result, vec![1, 2, 3]).unwrap();
    
    assert_eq!(filtered.total_hits(), 3);
}

#[test]
fn test_exclude_ids() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    for i in 1..=10 {
        engine.add_document(i, create_doc("Test", "Content")).unwrap();
    }
    
    let result = engine.search("test").unwrap();
    let excluded = engine.exclude_ids(&result, vec![1, 2, 3]).unwrap();
    
    assert_eq!(excluded.total_hits(), 7);
}

// ============================================
// Chinese Text Tests
// ============================================

#[test]
fn test_chinese_text_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("全文搜索引擎", "支持中文搜索")).unwrap();
    engine.add_document(2, create_doc("数据库系统", "高性能数据存储")).unwrap();
    
    let result = engine.search("搜索").unwrap();
    assert_eq!(result.total_hits(), 1);
    assert!(result.contains(1));
}

#[test]
fn test_mixed_language_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Rust全文搜索", "High performance search")).unwrap();
    
    // Search Chinese
    let result = engine.search("搜索").unwrap();
    assert_eq!(result.total_hits(), 1);
    
    // Search English
    let result = engine.search("rust").unwrap();
    assert_eq!(result.total_hits(), 1);
}

// ============================================
// Persistence Tests
// ============================================

#[test]
fn test_persistence_flush_and_reload() {
    let temp_dir = std::env::temp_dir();
    let index_path = temp_dir.join("test_persistence_reload.nfts");
    let index_str = index_path.to_str().unwrap().to_string();
    
    // Clean up
    let _ = std::fs::remove_file(&index_path);
    let _ = std::fs::remove_file(index_path.with_extension("nfts.wal"));
    
    // Create and populate
    {
        let engine = UnifiedEngine::new(
            EngineConfig::persistent(&index_str).with_drop_if_exists(true)
        ).unwrap();
        
        engine.add_document(1, create_doc("Persistent Doc", "Test data")).unwrap();
        engine.flush().unwrap();
    }
    
    // Reload and verify
    {
        let engine = UnifiedEngine::new(
            EngineConfig::persistent(&index_str)
        ).unwrap();
        
        let result = engine.search("persistent").unwrap();
        assert_eq!(result.total_hits(), 1);
    }
    
    // Clean up
    let _ = std::fs::remove_file(&index_path);
    let _ = std::fs::remove_file(index_path.with_extension("nfts.wal"));
}

// ============================================
// Statistics Tests
// ============================================

#[test]
fn test_engine_statistics() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Test", "Content")).unwrap();
    engine.search("test").unwrap();
    engine.search("test").unwrap(); // Cache hit
    
    let stats = engine.stats();
    
    assert!(stats.contains_key("search_count"));
    assert!(stats.contains_key("cache_hits"));
    assert!(stats.contains_key("term_count"));
    
    assert_eq!(*stats.get("search_count").unwrap() as u64, 2);
}

#[test]
fn test_fuzzy_config() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.set_fuzzy_config(0.8, 3, 30);
    
    let config = engine.get_fuzzy_config();
    assert_eq!(*config.get("threshold").unwrap(), 0.8);
    assert_eq!(*config.get("max_distance").unwrap() as usize, 3);
    assert_eq!(*config.get("max_candidates").unwrap() as usize, 30);
}

// ============================================
// Edge Cases Tests
// ============================================

#[test]
fn test_empty_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Test", "Content")).unwrap();
    
    let result = engine.search("nonexistent").unwrap();
    assert_eq!(result.total_hits(), 0);
    assert!(result.is_empty());
}

#[test]
fn test_empty_engine_search() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    let result = engine.search("anything").unwrap();
    assert_eq!(result.total_hits(), 0);
}

#[test]
fn test_large_batch_add() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    let docs: Vec<_> = (0..1000)
        .map(|i| (i as u32, create_doc(&format!("Doc {}", i), &format!("Content {}", i))))
        .collect();
    
    let count = engine.add_documents(docs).unwrap();
    assert_eq!(count, 1000);
    
    let result = engine.search("doc").unwrap();
    assert_eq!(result.total_hits(), 1000);
}

#[test]
fn test_special_characters() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Test@Email", "user@example.com")).unwrap();
    engine.add_document(2, create_doc("C++ Programming", "Learn C++")).unwrap();
    
    // Should still be searchable
    let result = engine.search("test").unwrap();
    assert!(result.total_hits() >= 1);
}

#[test]
fn test_cache_operations() {
    let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
    
    engine.add_document(1, create_doc("Test", "Content")).unwrap();
    
    // First search - cache miss
    engine.search("test").unwrap();
    
    // Second search - cache hit
    engine.search("test").unwrap();
    
    // Clear cache
    engine.clear_cache();
    
    // Third search - cache miss again
    engine.search("test").unwrap();
    
    let stats = engine.stats();
    assert!(*stats.get("search_count").unwrap() >= 3.0);
}

