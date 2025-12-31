//! High-Performance Search Engine Module
//! 
//! Implements parallel search, fuzzy search and ranking algorithms

use crate::bitmap::FastBitmap;
use crate::index::{InvertedIndex, IndexConfig};
use crate::simd_utils;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::cmp::Ordering as CmpOrdering;

/// Search result
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Document ID list
    pub doc_ids: Vec<u32>,
    /// Total hit count
    pub total_hits: u64,
    /// Search time (nanoseconds)
    pub elapsed_ns: u64,
    /// Whether fuzzy search was used
    pub fuzzy_used: bool,
}

impl SearchResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            doc_ids: Vec::new(),
            total_hits: 0,
            elapsed_ns: 0,
            fuzzy_used: false,
        }
    }
    
    /// Create from bitmap
    pub fn from_bitmap(bitmap: FastBitmap, elapsed_ns: u64, fuzzy_used: bool) -> Self {
        let total_hits = bitmap.len();
        Self {
            doc_ids: bitmap.to_vec(),
            total_hits,
            elapsed_ns,
            fuzzy_used,
        }
    }
}

/// Fuzzy search configuration
#[derive(Clone, Debug)]
pub struct FuzzyConfig {
    /// Similarity threshold (0.0 - 1.0)
    pub threshold: f64,
    /// Maximum edit distance
    pub max_distance: usize,
    /// Maximum candidate count
    pub max_candidates: usize,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            max_distance: 2,
            max_candidates: 20,
        }
    }
}

/// Search engine
pub struct SearchEngine {
    /// Inverted index
    index: Arc<InvertedIndex>,
    /// Fuzzy search configuration
    fuzzy_config: parking_lot::RwLock<FuzzyConfig>,
    /// Term cache (for fuzzy search)
    terms_cache: parking_lot::RwLock<Option<Vec<String>>>,
}

impl SearchEngine {
    /// Create search engine
    pub fn new(index_dir: Option<PathBuf>, config: IndexConfig) -> Self {
        Self {
            index: Arc::new(InvertedIndex::new(index_dir, config)),
            fuzzy_config: parking_lot::RwLock::new(FuzzyConfig::default()),
            terms_cache: parking_lot::RwLock::new(None),
        }
    }
    
    /// Create search engine with fuzzy configuration
    pub fn new_with_fuzzy(index_dir: Option<PathBuf>, config: IndexConfig, fuzzy_config: FuzzyConfig) -> Self {
        Self {
            index: Arc::new(InvertedIndex::new(index_dir, config)),
            fuzzy_config: parking_lot::RwLock::new(fuzzy_config),
            terms_cache: parking_lot::RwLock::new(None),
        }
    }
    
    /// Set fuzzy search configuration
    pub fn set_fuzzy_config(&self, config: FuzzyConfig) {
        let mut fuzzy_config = self.fuzzy_config.write();
        *fuzzy_config = config;
    }
    
    /// Get fuzzy search configuration
    pub fn get_fuzzy_config(&self) -> FuzzyConfig {
        self.fuzzy_config.read().clone()
    }
    
    /// Add document
    pub fn add_document(&self, doc_id: u32, fields: &[(&str, &str)]) {
        self.index.add_terms(doc_id, fields);
        
        // Clear term cache
        let mut cache = self.terms_cache.write();
        *cache = None;
        
        // Check if merge needed
        if self.index.should_merge() {
            let _ = self.index.merge_buffer();
        }
    }
    
    /// Batch add documents
    pub fn add_documents(&self, docs: &[(u32, Vec<(&str, &str)>)]) {
        for (doc_id, fields) in docs {
            self.index.add_terms(*doc_id, fields);
        }
        
        // Clear term cache
        let mut cache = self.terms_cache.write();
        *cache = None;
        
        // Check if merge needed
        if self.index.should_merge() {
            let _ = self.index.merge_buffer();
        }
    }
    
    /// Exact search
    pub fn search(&self, query: &str) -> SearchResult {
        let start = std::time::Instant::now();
        let bitmap = self.index.search(query);
        let elapsed = start.elapsed().as_nanos() as u64;
        
        SearchResult::from_bitmap(bitmap, elapsed, false)
    }
    
    /// Search and return bitmap (high-performance version)
    #[inline]
    pub fn search_bitmap(&self, query: &str) -> FastBitmap {
        self.index.search(query)
    }
    
    /// Fuzzy search
    pub fn fuzzy_search(&self, query: &str, min_results: usize) -> SearchResult {
        let start = std::time::Instant::now();
        
        // Try exact search first
        let exact_result = self.index.search(query);
        
        if exact_result.len() as usize >= min_results {
            let elapsed = start.elapsed().as_nanos() as u64;
            return SearchResult::from_bitmap(exact_result, elapsed, false);
        }
        
        // Perform fuzzy search
        let fuzzy_bitmap = self.perform_fuzzy_search(query);
        
        // Merge results
        let result = if exact_result.is_empty() {
            fuzzy_bitmap
        } else {
            exact_result.or(&fuzzy_bitmap)
        };
        
        let elapsed = start.elapsed().as_nanos() as u64;
        SearchResult::from_bitmap(result, elapsed, true)
    }
    
    /// Perform fuzzy search
    fn perform_fuzzy_search(&self, query: &str) -> FastBitmap {
        // Get similar terms
        let similar_terms = self.find_similar_terms(query);
        
        if similar_terms.is_empty() {
            return FastBitmap::new();
        }
        
        // Parallel search similar terms
        let bitmaps: Vec<FastBitmap> = similar_terms.par_iter()
            .filter_map(|(term, _score)| {
                let bitmap = self.index.search(term);
                if bitmap.is_empty() {
                    None
                } else {
                    Some(bitmap)
                }
            })
            .collect();
        
        // Merge all results
        let refs: Vec<_> = bitmaps.iter().collect();
        crate::bitmap::fast_union(&refs)
    }
    
    /// Find similar terms
    fn find_similar_terms(&self, query: &str) -> Vec<(String, f64)> {
        let query_len = query.chars().count();
        
        // Read fuzzy search configuration
        let fuzzy_config = self.fuzzy_config.read().clone();
        let max_distance = fuzzy_config.max_distance;
        let threshold = fuzzy_config.threshold;
        let max_candidates = fuzzy_config.max_candidates;
        
        // Get all terms
        // Note: In production, this should use more efficient data structures (like BK-Tree)
        let all_terms: Vec<String> = {
            let cache = self.terms_cache.read();
            if let Some(ref terms) = *cache {
                terms.clone()
            } else {
                drop(cache);
                // Collect terms from buffer
                let mut terms = Vec::new();
                // Need to get term list from index
                // Temporarily return empty
                terms
            }
        };
        
        // Parallel compute similarity
        let mut candidates: Vec<(String, f64)> = all_terms.par_iter()
            .filter_map(|term| {
                let term_len = term.chars().count();
                
                // Skip if length difference too large
                if (term_len as i32 - query_len as i32).abs() > max_distance as i32 {
                    return None;
                }
                
                // Fast pre-filter: check for common characters
                let query_chars: std::collections::HashSet<char> = query.chars().collect();
                let term_chars: std::collections::HashSet<char> = term.chars().collect();
                if query_chars.is_disjoint(&term_chars) {
                    return None;
                }
                
                // Calculate similarity
                let similarity = simd_utils::similarity_score(query, term);
                
                if similarity >= threshold {
                    Some((term.clone(), similarity))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
        
        // Return top N
        candidates.truncate(max_candidates);
        candidates
    }
    
    /// Remove document
    pub fn remove_document(&self, doc_id: u32) {
        self.index.remove_document(doc_id);
    }
    
    /// Batch remove documents
    pub fn remove_documents(&self, doc_ids: &[u32]) {
        for doc_id in doc_ids {
            self.index.remove_document(*doc_id);
        }
    }
    
    /// Update document (remove then add)
    pub fn update_document(&self, doc_id: u32, fields: &[(&str, &str)]) {
        self.remove_document(doc_id);
        self.add_document(doc_id, fields);
    }
    
    /// Batch update documents
    pub fn update_documents(&self, docs: &[(u32, Vec<(&str, &str)>)]) {
        // Batch remove first
        let doc_ids: Vec<u32> = docs.iter().map(|(id, _)| *id).collect();
        self.remove_documents(&doc_ids);
        
        // Then batch add
        self.add_documents(docs);
    }
    
    /// Save index
    pub fn save(&self) -> Result<(), crate::shard::ShardError> {
        self.index.save()
    }
    
    /// Load index
    pub fn load(&self) -> Result<(), crate::shard::ShardError> {
        self.index.load()
    }
    
    /// Flush buffer
    pub fn flush(&self) -> Result<(), crate::shard::ShardError> {
        self.index.merge_buffer()
    }
    
    /// Get index reference
    pub fn index(&self) -> &InvertedIndex {
        &self.index
    }
    
    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.index.buffer_size()
    }
    
    /// Get search statistics
    pub fn search_stats(&self) -> SearchStats {
        let stats = self.index.stats();
        SearchStats {
            total_searches: stats.search_count.load(std::sync::atomic::Ordering::Relaxed),
            avg_search_time_ns: stats.avg_search_time_ns.load(std::sync::atomic::Ordering::Relaxed),
            cache_hit_rate: self.index.stats().cache_hits.load(std::sync::atomic::Ordering::Relaxed) as f64 /
                (self.index.stats().cache_hits.load(std::sync::atomic::Ordering::Relaxed) + 
                 self.index.stats().cache_misses.load(std::sync::atomic::Ordering::Relaxed) + 1) as f64,
        }
    }
}

/// Search statistics
#[derive(Clone, Debug)]
pub struct SearchStats {
    pub total_searches: u64,
    pub avg_search_time_ns: u64,
    pub cache_hit_rate: f64,
}

/// Parallel searcher
/// 
/// For parallel search across multiple index shards
pub struct ParallelSearcher {
    engines: Vec<Arc<SearchEngine>>,
}

impl ParallelSearcher {
    /// Create parallel searcher
    pub fn new(engines: Vec<Arc<SearchEngine>>) -> Self {
        Self { engines }
    }
    
    /// Parallel search
    pub fn search(&self, query: &str) -> SearchResult {
        let start = std::time::Instant::now();
        
        // Parallel search all engines
        let bitmaps: Vec<FastBitmap> = self.engines.par_iter()
            .map(|engine| engine.search_bitmap(query))
            .collect();
        
        // Merge results
        let refs: Vec<_> = bitmaps.iter().collect();
        let merged = crate::bitmap::fast_union(&refs);
        
        let elapsed = start.elapsed().as_nanos() as u64;
        SearchResult::from_bitmap(merged, elapsed, false)
    }
    
    /// Parallel fuzzy search
    pub fn fuzzy_search(&self, query: &str, min_results: usize) -> SearchResult {
        let start = std::time::Instant::now();
        
        // Parallel search all engines
        let results: Vec<SearchResult> = self.engines.par_iter()
            .map(|engine| engine.fuzzy_search(query, min_results))
            .collect();
        
        // Merge all results
        let mut all_doc_ids = Vec::new();
        let mut fuzzy_used = false;
        
        for result in results {
            all_doc_ids.extend(result.doc_ids);
            fuzzy_used = fuzzy_used || result.fuzzy_used;
        }
        
        // Deduplicate
        all_doc_ids.sort_unstable();
        all_doc_ids.dedup();
        
        let elapsed = start.elapsed().as_nanos() as u64;
        
        SearchResult {
            total_hits: all_doc_ids.len() as u64,
            doc_ids: all_doc_ids,
            elapsed_ns: elapsed,
            fuzzy_used,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine() {
        let engine = SearchEngine::new(None, IndexConfig::default());
        
        engine.add_document(1, &[("title", "hello world")]);
        engine.add_document(2, &[("title", "hello rust")]);
        
        let result = engine.search("hello");
        assert_eq!(result.total_hits, 2);
    }

    #[test]
    fn test_chinese_search() {
        let engine = SearchEngine::new(None, IndexConfig::default());
        
        engine.add_document(1, &[("content", "全文搜索引擎")]);
        engine.add_document(2, &[("content", "高性能搜索")]);
        
        let result = engine.search("搜索");
        assert!(result.total_hits >= 1);
    }
}
