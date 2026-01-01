//! Inverted Index Core Module
//! 
//! Implements high-performance inverted index storage and querying

use crate::bitmap::FastBitmap;
use crate::cache::ShardedCache;
use crate::shard::{ShardManager, ShardWriter, ShardError};
use dashmap::DashMap;
use rustc_hash::FxHashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use regex::Regex;

/// Index configuration
#[derive(Clone, Debug)]
pub struct IndexConfig {
    /// Maximum Chinese n-gram length
    pub max_chinese_length: usize,
    /// Minimum term length
    pub min_term_length: usize,
    /// Shard bits
    pub shard_bits: u32,
    /// Cache size
    pub cache_size: usize,
    /// Buffer size
    pub buffer_size: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_chinese_length: 4,
            min_term_length: 2,
            shard_bits: 8,
            cache_size: 10000,
            buffer_size: 100000,
        }
    }
}

/// High-performance inverted index
pub struct InvertedIndex {
    /// Index directory
    index_dir: Option<PathBuf>,
    /// Configuration
    config: IndexConfig,
    /// Memory buffer
    buffer: DashMap<String, FastBitmap>,
    /// Shard manager
    shard_manager: Option<ShardManager>,
    /// Bitmap cache
    bitmap_cache: ShardedCache<String, FastBitmap>,
    /// Chinese regex pattern
    chinese_pattern: Regex,
    /// Chinese substring cache
    chinese_cache: DashMap<String, Vec<String>>,
    /// Statistics
    stats: IndexStats,
}

/// Index statistics
#[derive(Default)]
pub struct IndexStats {
    pub total_terms: AtomicU64,
    pub total_docs: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub search_count: AtomicU64,
    pub avg_search_time_ns: AtomicU64,
}

impl InvertedIndex {
    /// Create new inverted index
    pub fn new(index_dir: Option<PathBuf>, config: IndexConfig) -> Self {
        let shard_manager = index_dir.as_ref().map(|dir| {
            let shards_dir = dir.join("shards");
            std::fs::create_dir_all(&shards_dir).ok();
            ShardManager::new(shards_dir, config.shard_bits)
        });
        
        Self {
            index_dir,
            config: config.clone(),
            buffer: DashMap::new(),
            shard_manager,
            bitmap_cache: ShardedCache::new(config.cache_size, 16),
            chinese_pattern: Regex::new(r"[\u4e00-\u9fff]+").unwrap(),
            chinese_cache: DashMap::new(),
            stats: IndexStats::default(),
        }
    }
    
    /// Add document terms
    pub fn add_terms(&self, doc_id: u32, terms: &[(&str, &str)]) {
        for (_field, value) in terms {
            let value_lower = value.to_lowercase();
            
            // Process complete field
            if value_lower.len() >= self.config.min_term_length {
                self.add_to_buffer(&value_lower, doc_id);
            }
            
            // Process Chinese
            for cap in self.chinese_pattern.find_iter(&value_lower) {
                let seg = cap.as_str();
                let substrings = self.get_chinese_substrings(seg);
                for substr in substrings {
                    self.add_to_buffer(&substr, doc_id);
                }
            }
            
            // Process English words
            for word in value_lower.split_whitespace() {
                if word.len() >= self.config.min_term_length {
                    // Check if pure English
                    if word.chars().all(|c| c.is_ascii_alphabetic()) {
                        self.add_to_buffer(word, doc_id);
                    }
                }
            }
            
            // Process phrases
            if value_lower.contains(' ') {
                self.add_to_buffer(&value_lower, doc_id);
            }
        }
        
        self.stats.total_docs.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add to buffer
    #[inline]
    fn add_to_buffer(&self, term: &str, doc_id: u32) {
        self.buffer
            .entry(term.to_string())
            .or_insert_with(FastBitmap::new)
            .add(doc_id);
    }
    
    /// Get Chinese substrings
    fn get_chinese_substrings(&self, seg: &str) -> Vec<String> {
        // Check cache first
        if let Some(cached) = self.chinese_cache.get(seg) {
            return cached.clone();
        }
        
        let chars: Vec<char> = seg.chars().collect();
        let n = chars.len();
        let mut substrings = Vec::new();
        
        for length in self.config.min_term_length..=self.config.max_chinese_length.min(n) {
            for j in 0..=(n - length) {
                let substr: String = chars[j..j + length].iter().collect();
                substrings.push(substr);
            }
        }
        
        // Cache result
        self.chinese_cache.insert(seg.to_string(), substrings.clone());
        
        substrings
    }
    
    /// Search
    pub fn search(&self, query: &str) -> FastBitmap {
        let start = std::time::Instant::now();
        self.stats.search_count.fetch_add(1, Ordering::Relaxed);
        
        let query = query.trim().to_lowercase();
        if query.is_empty() {
            return FastBitmap::new();
        }
        
        let result = if query.contains(' ') {
            // Phrase search
            self.search_phrase(&query)
        } else if self.chinese_pattern.is_match(&query) {
            // Chinese search
            self.search_chinese(&query)
        } else {
            // Single word search
            self.search_term(&query)
        };
        
        let elapsed = start.elapsed().as_nanos() as u64;
        let prev_avg = self.stats.avg_search_time_ns.load(Ordering::Relaxed);
        let count = self.stats.search_count.load(Ordering::Relaxed);
        let new_avg = (prev_avg * (count - 1) + elapsed) / count;
        self.stats.avg_search_time_ns.store(new_avg, Ordering::Relaxed);
        
        result
    }
    
    /// Search single term
    fn search_term(&self, term: &str) -> FastBitmap {
        self.get_bitmap(term).unwrap_or_default()
    }
    
    /// Search phrase
    fn search_phrase(&self, phrase: &str) -> FastBitmap {
        let words: Vec<&str> = phrase.split_whitespace()
            .filter(|w| w.len() >= self.config.min_term_length)
            .collect();
        
        if words.is_empty() {
            return FastBitmap::new();
        }
        
        // Get all term bitmaps
        let bitmaps: Vec<_> = words.iter()
            .filter_map(|word| {
                if self.chinese_pattern.is_match(word) {
                    Some(self.search_chinese(word))
                } else {
                    self.get_bitmap(word)
                }
            })
            .collect();
        
        if bitmaps.is_empty() {
            return FastBitmap::new();
        }
        
        // Fast intersection
        let refs: Vec<_> = bitmaps.iter().collect();
        crate::bitmap::fast_intersection(&refs)
    }
    
    /// Search Chinese
    fn search_chinese(&self, query: &str) -> FastBitmap {
        let chars: Vec<char> = query.chars().collect();
        let n = chars.len();
        
        if n < self.config.min_term_length {
            return FastBitmap::new();
        }
        
        // Try longest match
        let max_len = n.min(self.config.max_chinese_length);
        
        for length in (self.config.min_term_length..=max_len).rev() {
            for i in 0..=(n - length) {
                let substr: String = chars[i..i + length].iter().collect();
                if let Some(bitmap) = self.get_bitmap(&substr) {
                    if bitmap.len() < 1000 {
                        return bitmap;
                    }
                    
                    // Try intersection with adjacent substrings
                    if i > 0 && i + length <= n {
                        let prev: String = chars[i - 1..i - 1 + length].iter().collect();
                        if let Some(prev_bitmap) = self.get_bitmap(&prev) {
                            let intersection = bitmap.and(&prev_bitmap);
                            if !intersection.is_empty() {
                                return intersection;
                            }
                        }
                    }
                    
                    return bitmap;
                }
            }
        }
        
        FastBitmap::new()
    }
    
    /// Get bitmap (with cache)
    fn get_bitmap(&self, term: &str) -> Option<FastBitmap> {
        // 1. Check cache first
        if let Some(bitmap) = self.bitmap_cache.get(&term.to_string()) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Some(bitmap);
        }
        
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // 2. Check buffer
        if let Some(bitmap) = self.buffer.get(term) {
            let result = bitmap.clone();
            self.bitmap_cache.insert(term.to_string(), result.clone());
            return Some(result);
        }
        
        // 3. Load from shard
        if let Some(ref manager) = self.shard_manager {
            if let Some(bitmap) = manager.get_bitmap(term) {
                self.bitmap_cache.insert(term.to_string(), bitmap.clone());
                return Some(bitmap);
            }
        }
        
        None
    }
    
    /// Merge buffer to shards
    pub fn merge_buffer(&self) -> Result<(), ShardError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        
        let index_dir = match &self.index_dir {
            Some(dir) => dir,
            None => return Ok(()), // Memory mode
        };
        
        // Group by shard
        let shard_manager = self.shard_manager.as_ref().unwrap();
        let mut shard_buffers: FxHashMap<u32, FxHashMap<String, FastBitmap>> = FxHashMap::default();
        
        for entry in self.buffer.iter() {
            let term = entry.key().clone();
            let bitmap = entry.value().clone();
            let shard_id = shard_manager.get_shard_id(&term);
            
            shard_buffers
                .entry(shard_id)
                .or_default()
                .insert(term, bitmap);
        }
        
        // Write to shards
        let shards_dir = index_dir.join("shards");
        for (shard_id, terms) in shard_buffers.iter() {
            let shard_path = shards_dir.join(format!("shard_{}.nfts", shard_id));
            let mut writer = ShardWriter::new(&shard_path);
            
            // Load existing data first
            if shard_path.exists() {
                if let Some(existing_shard) = shard_manager.get_shard(*shard_id) {
                    for term in existing_shard.terms() {
                        if !terms.contains_key(term) {
                            if let Some(bitmap) = existing_shard.get_bitmap(term) {
                                writer.add_term(term.clone(), &bitmap)?;
                            }
                        }
                    }
                }
            }
            
            // Add/merge new data
            for (term, bitmap) in terms {
                writer.merge_term(term.clone(), bitmap)?;
            }
            
            writer.write()?;
        }
        
        // Clear buffer
        self.buffer.clear();
        self.bitmap_cache.clear();
        
        Ok(())
    }
    
    /// Save index
    pub fn save(&self) -> Result<(), ShardError> {
        self.merge_buffer()
    }
    
    /// Load index
    pub fn load(&self) -> Result<(), ShardError> {
        if let Some(ref manager) = self.shard_manager {
            manager.reload()?;
        }
        Ok(())
    }
    
    /// Remove document
    pub fn remove_document(&self, doc_id: u32) {
        // Remove from buffer
        for mut entry in self.buffer.iter_mut() {
            entry.value_mut().remove(doc_id);
        }
        
        // Clear cache
        self.bitmap_cache.clear();
    }
    
    /// Get statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }
    
    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if merge needed
    pub fn should_merge(&self) -> bool {
        self.buffer.len() >= self.config.buffer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_indexing() {
        let index = InvertedIndex::new(None, IndexConfig::default());
        
        index.add_terms(1, &[("title", "hello world")]);
        index.add_terms(2, &[("title", "hello rust")]);
        index.add_terms(3, &[("title", "goodbye world")]);
        
        let result = index.search("hello");
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(2));
    }

    #[test]
    fn test_chinese_search() {
        let index = InvertedIndex::new(None, IndexConfig::default());
        
        index.add_terms(1, &[("content", "这是一个测试文档")]);
        index.add_terms(2, &[("content", "另一个测试内容")]);
        index.add_terms(3, &[("content", "完全不同的东西")]);
        
        let result = index.search("测试");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_phrase_search() {
        let index = InvertedIndex::new(None, IndexConfig::default());
        
        index.add_terms(1, &[("title", "hello world")]);
        index.add_terms(2, &[("title", "hello rust world")]);
        index.add_terms(3, &[("title", "goodbye world")]);
        
        let result = index.search("hello world");
        assert!(result.len() >= 1);
    }
}
