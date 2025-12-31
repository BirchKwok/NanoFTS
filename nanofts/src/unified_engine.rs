//! Unified Search Engine - Single-File LSM Implementation
//!
//! Replacement for PagedEngine, LsmEngine, LsmSingleEngine
//! Supports: memory-only mode, single-file persistence, fuzzy search, document delete/update

use crate::bitmap::FastBitmap;
use crate::lsm_single::LsmSingleIndex;
use crate::simd_utils;
use dashmap::DashMap;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Search result handle
#[pyclass]
#[derive(Clone)]
pub struct ResultHandle {
    bitmap: Arc<FastBitmap>,
    query: String,
    elapsed_ns: u64,
    fuzzy_used: bool,
}

#[pymethods]
impl ResultHandle {
    #[getter]
    fn total_hits(&self) -> u64 {
        self.bitmap.len()
    }
    
    #[getter]
    fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }
    
    #[getter]
    fn fuzzy_used(&self) -> bool {
        self.fuzzy_used
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000.0
    }
    
    fn elapsed_us(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000.0
    }
    
    fn contains(&self, doc_id: u32) -> bool {
        self.bitmap.contains(doc_id)
    }
    
    fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }
    
    #[pyo3(signature = (n=100))]
    fn top(&self, n: usize) -> Vec<u32> {
        self.bitmap.iter().take(n).collect()
    }
    
    fn to_list(&self) -> Vec<u32> {
        self.bitmap.to_vec()
    }
    
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<u32>>> {
        let ids: Vec<u32> = self.bitmap.to_vec();
        Ok(numpy::PyArray1::from_vec_bound(py, ids))
    }
    
    fn page(&self, offset: usize, limit: usize) -> Vec<u32> {
        self.bitmap.iter().skip(offset).take(limit).collect()
    }
    
    fn intersect(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.and(&other.bitmap)),
            query: format!("({}) AND ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used || other.fuzzy_used,
        }
    }
    
    fn union(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.or(&other.bitmap)),
            query: format!("({}) OR ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used || other.fuzzy_used,
        }
    }
    
    fn difference(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.andnot(&other.bitmap)),
            query: format!("({}) NOT ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used,
        }
    }
    
    fn __len__(&self) -> usize {
        self.bitmap.len() as usize
    }
    
    fn __repr__(&self) -> String {
        format!(
            "ResultHandle(hits={}, query='{}', elapsed={:.3}ms)",
            self.bitmap.len(), self.query, self.elapsed_ns as f64 / 1_000_000.0
        )
    }
}

/// Fuzzy search configuration
#[pyclass]
#[derive(Clone)]
pub struct FuzzyConfig {
    #[pyo3(get, set)]
    pub threshold: f64,
    #[pyo3(get, set)]
    pub max_distance: usize,
    #[pyo3(get, set)]
    pub max_candidates: usize,
}

#[pymethods]
impl FuzzyConfig {
    #[new]
    #[pyo3(signature = (threshold=0.7, max_distance=2, max_candidates=20))]
    fn new(threshold: f64, max_distance: usize, max_candidates: usize) -> Self {
        Self { threshold, max_distance, max_candidates }
    }
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self { threshold: 0.7, max_distance: 2, max_candidates: 20 }
    }
}

/// Engine statistics
struct EngineStats {
    search_count: AtomicU64,
    fuzzy_search_count: AtomicU64,
    cache_hits: AtomicU64,
    total_search_ns: AtomicU64,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            search_count: AtomicU64::new(0),
            fuzzy_search_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            total_search_ns: AtomicU64::new(0),
        }
    }
}

/// Unified Search Engine
/// 
/// Single-file LSM implementation supporting all features:
/// - Memory-only mode (empty index_file)
/// - Single-file persistence (.nfts format)
/// - Lazy load mode (large file, low memory)
/// - Fuzzy search
/// - Document delete/update
#[pyclass(subclass)]
pub struct UnifiedEngine {
    // Storage layer
    index: Option<Arc<LsmSingleIndex>>,
    // Memory buffer (memory-only mode or write buffer)
    buffer: DashMap<String, FastBitmap>,
    // Document-term mapping (for efficient deletion)
    doc_terms: DashMap<u32, Vec<String>>,
    track_doc_terms: bool,
    // Deletion markers (tombstone)
    deleted_docs: DashMap<u32, ()>,
    // Updated documents (data in index should be ignored, use buffer data only)
    updated_docs: DashMap<u32, ()>,
    // Configuration
    max_chinese_length: usize,
    min_term_length: usize,
    fuzzy_config: RwLock<FuzzyConfig>,
    // Cache
    result_cache: DashMap<String, Arc<FastBitmap>>,
    cache_enabled: bool,
    // Statistics
    stats: EngineStats,
    // Mode
    memory_only: bool,
    lazy_load: bool,
    preloaded: std::sync::atomic::AtomicBool,
    // Regex
    chinese_pattern: regex::Regex,
    // Compact lock: compact gets write lock, flush gets read lock
    compact_lock: RwLock<()>,
}

#[pymethods]
impl UnifiedEngine {
    /// Create unified search engine
    /// 
    /// Args:
    ///     index_file: Index file path, empty string for memory-only mode
    ///     max_chinese_length: Maximum Chinese n-gram length
    ///     min_term_length: Minimum term length
    ///     fuzzy_threshold: Fuzzy search similarity threshold
    ///     fuzzy_max_distance: Fuzzy search maximum edit distance
    ///     track_doc_terms: Whether to track document terms (for efficient deletion)
    ///     drop_if_exists: Whether to delete existing index file
    ///     lazy_load: Whether to enable lazy load mode (large file, low memory)
    ///     cache_size: LRU cache size in lazy load mode
    #[new]
    #[pyo3(signature = (
        index_file="".to_string(),
        max_chinese_length=4,
        min_term_length=2,
        fuzzy_threshold=0.7,
        fuzzy_max_distance=2,
        track_doc_terms=false,
        drop_if_exists=false,
        lazy_load=false,
        cache_size=10000
    ))]
    fn new(
        index_file: String,
        max_chinese_length: usize,
        min_term_length: usize,
        fuzzy_threshold: f64,
        fuzzy_max_distance: usize,
        track_doc_terms: bool,
        drop_if_exists: bool,
        lazy_load: bool,
        cache_size: usize,
    ) -> PyResult<Self> {
        let memory_only = index_file.is_empty();
        
        let index = if memory_only {
            None
        } else {
            let path = std::path::PathBuf::from(&index_file);
            
            let idx = if path.exists() && drop_if_exists {
                std::fs::remove_file(&path)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                LsmSingleIndex::create_full_options(&path, true, lazy_load, cache_size)
            } else if path.exists() {
                LsmSingleIndex::open_full_options(&path, true, lazy_load, cache_size)
            } else {
                LsmSingleIndex::create_full_options(&path, true, lazy_load, cache_size)
            }.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            Some(Arc::new(idx))
        };
        
        Ok(Self {
            index,
            buffer: DashMap::new(),
            doc_terms: DashMap::new(),
            track_doc_terms,
            deleted_docs: DashMap::new(),
            updated_docs: DashMap::new(),
            max_chinese_length,
            min_term_length,
            fuzzy_config: RwLock::new(FuzzyConfig {
                threshold: fuzzy_threshold,
                max_distance: fuzzy_max_distance,
                max_candidates: 20,
            }),
            result_cache: DashMap::new(),
            cache_enabled: true,
            stats: EngineStats::default(),
            memory_only,
            lazy_load,
            preloaded: std::sync::atomic::AtomicBool::new(false),
            chinese_pattern: regex::Regex::new(r"[\u4e00-\u9fff]+").unwrap(),
            compact_lock: RwLock::new(()),
        })
    }
    
    // ==================== Document Operations ====================
    
    /// Add single document
    fn add_document(&self, doc_id: u32, fields: HashMap<String, String>) -> PyResult<()> {
        // If previously deleted, remove from deleted_docs
        self.deleted_docs.remove(&doc_id);
        // Also remove from updated_docs (may have been updated before)
        self.updated_docs.remove(&doc_id);
        
        let text: String = fields.values().cloned().collect::<Vec<_>>().join(" ");
        self.add_text(doc_id, &text);
        
        // Clear result cache
        self.result_cache.clear();
        Ok(())
    }
    
    /// Batch add documents
    fn add_documents(&self, docs: Vec<(u32, HashMap<String, String>)>) -> PyResult<usize> {
        let count = docs.len();
        for (doc_id, fields) in docs {
            // If previously deleted, remove from deleted_docs
            self.deleted_docs.remove(&doc_id);
            // Also remove from updated_docs
            self.updated_docs.remove(&doc_id);
            
            let text: String = fields.values().cloned().collect::<Vec<_>>().join(" ");
            self.add_text(doc_id, &text);
        }
        self.result_cache.clear();
        Ok(count)
    }
    
    /// Update document
    /// 
    /// Update document content. After update:
    /// - Old content no longer searchable (removed from buffer, index data marked ignored)
    /// - New content is searchable (in buffer)
    /// - Call compact() to actually remove old data from index
    fn update_document(&self, doc_id: u32, fields: HashMap<String, String>) -> PyResult<()> {
        // 1. Mark document as "updated" (ignore old data in index during search)
        self.updated_docs.insert(doc_id, ());
        
        // 2. If track_doc_terms enabled, remove old terms from buffer
        if self.track_doc_terms {
            if let Some((_, terms)) = self.doc_terms.remove(&doc_id) {
                for term in terms {
                    if let Some(mut entry) = self.buffer.get_mut(&term) {
                        entry.remove(doc_id);
                    }
                }
            }
        }
        
        // 3. Ensure document not in deleted_docs (if previously deleted)
        self.deleted_docs.remove(&doc_id);
        
        // 4. Add new content to buffer
        let text: String = fields.values().cloned().collect::<Vec<_>>().join(" ");
        self.add_text(doc_id, &text);
        
        // 5. Clear result cache
        self.result_cache.clear();
        
        Ok(())
    }
    
    /// Delete document
    /// 
    /// Uses tombstone mechanism, deleted documents automatically excluded from search
    fn remove_document(&self, doc_id: u32) -> PyResult<()> {
        // Add tombstone marker
        self.deleted_docs.insert(doc_id, ());
        
        // If track_doc_terms enabled, also remove from buffer (improve memory efficiency)
        if self.track_doc_terms {
            if let Some((_, terms)) = self.doc_terms.remove(&doc_id) {
                for term in terms {
                    if let Some(mut entry) = self.buffer.get_mut(&term) {
                        entry.remove(doc_id);
                    }
                }
            }
        }
        
        self.result_cache.clear();
        Ok(())
    }
    
    /// Batch delete documents
    fn remove_documents(&self, doc_ids: Vec<u32>) -> PyResult<()> {
        for doc_id in doc_ids {
            self.remove_document(doc_id)?;
        }
        Ok(())
    }
    
    // ==================== Search Operations ====================
    
    /// Search
    fn search(&self, query: &str) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        self.stats.search_count.fetch_add(1, Ordering::Relaxed);
        
        // Check cache
        if self.cache_enabled {
            if let Some(cached) = self.result_cache.get(query) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(ResultHandle {
                    bitmap: cached.clone(),
                    query: query.to_string(),
                    elapsed_ns: start.elapsed().as_nanos() as u64,
                    fuzzy_used: false,
                });
            }
        }
        
        let bitmap = self.search_internal(query);
        let bitmap = Arc::new(bitmap);
        
        if self.cache_enabled {
            self.result_cache.insert(query.to_string(), bitmap.clone());
        }
        
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats.total_search_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        
        Ok(ResultHandle {
            bitmap,
            query: query.to_string(),
            elapsed_ns,
            fuzzy_used: false,
        })
    }
    
    /// Fuzzy search
    #[pyo3(signature = (query, min_results=5))]
    fn fuzzy_search(&self, query: &str, min_results: usize) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        self.stats.fuzzy_search_count.fetch_add(1, Ordering::Relaxed);
        
        // Try exact search first
        let exact_result = self.search_internal(query);
        if exact_result.len() >= min_results as u64 {
            return Ok(ResultHandle {
                bitmap: Arc::new(exact_result),
                query: query.to_string(),
                elapsed_ns: start.elapsed().as_nanos() as u64,
                fuzzy_used: false,
            });
        }
        
        // Fuzzy search
        let bitmap = self.fuzzy_search_internal(query);
        
        Ok(ResultHandle {
            bitmap: Arc::new(bitmap),
            query: query.to_string(),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: true,
        })
    }
    
    /// Batch search
    fn search_batch(&self, queries: Vec<String>) -> PyResult<Vec<ResultHandle>> {
        let results: Vec<ResultHandle> = queries.par_iter()
            .map(|q| {
                let start = std::time::Instant::now();
                let bitmap = Arc::new(self.search_internal(q));
                ResultHandle {
                    bitmap,
                    query: q.clone(),
                    elapsed_ns: start.elapsed().as_nanos() as u64,
                    fuzzy_used: false,
                }
            })
            .collect();
        Ok(results)
    }
    
    /// AND search
    fn search_and(&self, queries: Vec<String>) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        
        let mut result: Option<FastBitmap> = None;
        for query in &queries {
            let bitmap = self.search_internal(query);
            result = Some(match result {
                Some(mut r) => { r.and_inplace(&bitmap); r }
                None => bitmap,
            });
            if result.as_ref().map_or(true, |r| r.is_empty()) {
                break;
            }
        }
        
        Ok(ResultHandle {
            bitmap: Arc::new(result.unwrap_or_else(FastBitmap::new)),
            query: queries.join(" AND "),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: false,
        })
    }
    
    /// OR search
    fn search_or(&self, queries: Vec<String>) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        
        let mut result = FastBitmap::new();
        for query in &queries {
            let bitmap = self.search_internal(query);
            result.or_inplace(&bitmap);
        }
        
        Ok(ResultHandle {
            bitmap: Arc::new(result),
            query: queries.join(" OR "),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: false,
        })
    }
    
    /// Filter results
    fn filter_by_ids(&self, result: &ResultHandle, allowed_ids: Vec<u32>) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        let filter = FastBitmap::from_iter(allowed_ids);
        let filtered = result.bitmap.and(&filter);
        
        Ok(ResultHandle {
            bitmap: Arc::new(filtered),
            query: format!("{} [filtered]", result.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: result.fuzzy_used,
        })
    }
    
    /// Exclude IDs
    fn exclude_ids(&self, result: &ResultHandle, excluded_ids: Vec<u32>) -> PyResult<ResultHandle> {
        let start = std::time::Instant::now();
        let exclude = FastBitmap::from_iter(excluded_ids);
        let filtered = result.bitmap.andnot(&exclude);
        
        Ok(ResultHandle {
            bitmap: Arc::new(filtered),
            query: format!("{} [excluded]", result.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: result.fuzzy_used,
        })
    }
    
    // ==================== Configuration Operations ====================
    
    #[pyo3(signature = (threshold=0.7, max_distance=2, max_candidates=20))]
    fn set_fuzzy_config(&self, threshold: f64, max_distance: usize, max_candidates: usize) -> PyResult<()> {
        let mut config = self.fuzzy_config.write();
        config.threshold = threshold;
        config.max_distance = max_distance;
        config.max_candidates = max_candidates;
        Ok(())
    }
    
    fn get_fuzzy_config(&self) -> HashMap<String, f64> {
        let config = self.fuzzy_config.read();
        let mut map = HashMap::new();
        map.insert("threshold".to_string(), config.threshold);
        map.insert("max_distance".to_string(), config.max_distance as f64);
        map.insert("max_candidates".to_string(), config.max_candidates as f64);
        map
    }
    
    // ==================== Persistence Operations ====================
    
    /// Flush to disk
    /// 
    /// Persist memory buffer data to index file
    /// After successful flush, buffer is cleared to avoid duplicate writes
    fn flush(&self) -> PyResult<usize> {
        if self.memory_only {
            return Ok(0);
        }
        
        // Get read lock to prevent flush during compact
        let _lock = self.compact_lock.read();
        
        if let Some(ref index) = self.index {
            // Collect all doc_ids and data from buffer (these documents' new data will be written to index)
            // Note: use remove instead of iter to avoid concurrency issues
            let mut flushed_doc_ids = std::collections::HashSet::new();
            let mut entries: Vec<(String, FastBitmap)> = Vec::new();
            
            // Remove all entries from buffer (atomic operation)
            let keys: Vec<String> = self.buffer.iter().map(|e| e.key().clone()).collect();
            for key in keys {
                if let Some((term, bitmap)) = self.buffer.remove(&key) {
                    for doc_id in bitmap.iter() {
                        flushed_doc_ids.insert(doc_id);
                    }
                    entries.push((term, bitmap));
                }
            }
            
            let count = entries.len();
            
            if !entries.is_empty() {
                index.upsert_batch(entries);
                index.flush()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                
                // For doc_ids written in this flush, remove from updated_docs
                // So subsequent searches correctly return their new data
                for doc_id in flushed_doc_ids {
                    self.updated_docs.remove(&doc_id);
                }
            }
            
            Ok(count)
        } else {
            Ok(0)
        }
    }
    
    /// Save (same as flush)
    fn save(&self) -> PyResult<usize> {
        self.flush()
    }
    
    /// Load (no-op, data is loaded on open)
    fn load(&self) -> PyResult<()> {
        self.result_cache.clear();
        Ok(())
    }
    
    /// Preload
    fn preload(&self) -> PyResult<usize> {
        if self.memory_only || self.preloaded.load(Ordering::Relaxed) {
            return Ok(0);
        }
        
        if let Some(ref index) = self.index {
            // Data is already loaded in index, mark as preloaded
            self.preloaded.store(true, Ordering::Relaxed);
            Ok(index.term_count())
        } else {
            Ok(0)
        }
    }
    
    /// Compact (also applies deletions, making them persistent)
    fn compact(&self) -> PyResult<()> {
        if let Some(ref index) = self.index {
            // Get write lock to block other flush operations
            let _lock = self.compact_lock.write();
            
            // Step 1: Save buffer data but don't flush immediately
            // For updated documents, we need to delete old data first
            let mut pending_entries: Vec<(String, FastBitmap)> = Vec::new();
            let mut flushed_doc_ids = std::collections::HashSet::new();
            {
                let keys: Vec<String> = self.buffer.iter().map(|e| e.key().clone()).collect();
                for key in keys {
                    if let Some((term, bitmap)) = self.buffer.remove(&key) {
                        for doc_id in bitmap.iter() {
                            flushed_doc_ids.insert(doc_id);
                        }
                        pending_entries.push((term, bitmap));
                    }
                }
            }
            
            // Step 2: Collect doc_ids to delete (including actually deleted and updated)
            // Updated documents need old term associations removed from index
            let deleted: Vec<u32> = self.deleted_docs.iter()
                .map(|e| *e.key())
                .chain(self.updated_docs.iter().map(|e| *e.key()))
                .collect();
            
            // Step 3: Execute compact and apply deletions (removes old data)
            index.compact_with_deletions(&deleted)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            // Step 4: Now flush new data (updated document content)
            if !pending_entries.is_empty() {
                index.upsert_batch(pending_entries);
                index.flush()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            
            // Clean up state
            for doc_id in flushed_doc_ids {
                self.updated_docs.remove(&doc_id);
            }
            
            // After compact, process data added to buffer during compact
            // Note: other threads may have added data to buffer during compact_with_deletions
            {
                let mut flushed_doc_ids = std::collections::HashSet::new();
                let mut entries: Vec<(String, FastBitmap)> = Vec::new();
                
                let keys: Vec<String> = self.buffer.iter().map(|e| e.key().clone()).collect();
                for key in keys {
                    if let Some((term, bitmap)) = self.buffer.remove(&key) {
                        for doc_id in bitmap.iter() {
                            flushed_doc_ids.insert(doc_id);
                        }
                        entries.push((term, bitmap));
                    }
                }
                
                if !entries.is_empty() {
                    index.upsert_batch(entries);
                    index.flush()
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    
                    for doc_id in flushed_doc_ids {
                        self.updated_docs.remove(&doc_id);
                    }
                }
            }
            
            // Clear tombstones and update markers (persisted now)
            self.deleted_docs.clear();
            self.updated_docs.clear();
        }
        Ok(())
    }
    
    // ==================== Cache Operations ====================
    
    fn clear_cache(&self) {
        self.result_cache.clear();
    }
    
    fn clear_doc_terms(&self) {
        self.doc_terms.clear();
    }
    
    // ==================== Statistics Operations ====================
    
    fn is_memory_only(&self) -> bool {
        self.memory_only
    }
    
    fn term_count(&self) -> usize {
        let buffer_count = self.buffer.len();
        let index_count = self.index.as_ref().map_or(0, |i| i.term_count());
        buffer_count + index_count
    }
    
    fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    
    fn doc_terms_size(&self) -> usize {
        self.doc_terms.len()
    }
    
    fn page_count(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.segment_count())
    }
    
    fn segment_count(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.segment_count())
    }
    
    fn memtable_size(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.memtable_size())
    }
    
    fn stats(&self) -> HashMap<String, f64> {
        let search_count = self.stats.search_count.load(Ordering::Relaxed);
        let cache_hits = self.stats.cache_hits.load(Ordering::Relaxed);
        let total_ns = self.stats.total_search_ns.load(Ordering::Relaxed);
        
        let mut map = HashMap::new();
        map.insert("search_count".to_string(), search_count as f64);
        map.insert("fuzzy_search_count".to_string(), 
            self.stats.fuzzy_search_count.load(Ordering::Relaxed) as f64);
        map.insert("cache_hits".to_string(), cache_hits as f64);
        map.insert("cache_hit_rate".to_string(), 
            if search_count > 0 { cache_hits as f64 / search_count as f64 } else { 0.0 });
        map.insert("avg_search_us".to_string(),
            if search_count > 0 { total_ns as f64 / search_count as f64 / 1000.0 } else { 0.0 });
        map.insert("result_cache_size".to_string(), self.result_cache.len() as f64);
        map.insert("buffer_size".to_string(), self.buffer.len() as f64);
        map.insert("term_count".to_string(), self.term_count() as f64);
        map.insert("deleted_count".to_string(), self.deleted_docs.len() as f64);
        map.insert("memory_only".to_string(), if self.memory_only { 1.0 } else { 0.0 });
        map.insert("lazy_load".to_string(), if self.lazy_load { 1.0 } else { 0.0 });
        map.insert("track_doc_terms".to_string(), if self.track_doc_terms { 1.0 } else { 0.0 });
        
        // WAL and lazy load statistics
        if let Some(ref index) = self.index {
            map.insert("wal_enabled".to_string(), if index.is_wal_enabled() { 1.0 } else { 0.0 });
            map.insert("wal_size".to_string(), index.wal_size() as f64);
            map.insert("wal_pending_batches".to_string(), index.wal_pending_batches() as f64);
            
            // Lazy load cache statistics
            if index.is_lazy_load() {
                let (lru_hits, lru_misses, lru_size) = index.cache_stats();
                map.insert("lru_cache_hits".to_string(), lru_hits as f64);
                map.insert("lru_cache_misses".to_string(), lru_misses as f64);
                map.insert("lru_cache_size".to_string(), lru_size as f64);
                map.insert("lru_cache_hit_rate".to_string(), index.cache_hit_rate());
            }
        }
        
        map
    }
    
    /// Get deleted document count
    fn deleted_count(&self) -> usize {
        self.deleted_docs.len()
    }
    
    /// Whether lazy load is enabled
    fn is_lazy_load(&self) -> bool {
        self.lazy_load
    }
    
    /// Warmup cache (load specified terms into cache in lazy load mode)
    fn warmup_terms(&self, terms: Vec<String>) -> usize {
        if let Some(ref index) = self.index {
            index.warmup(&terms)
        } else {
            0
        }
    }
    
    /// Clear LRU cache (lazy load mode)
    fn clear_lru_cache(&self) {
        if let Some(ref index) = self.index {
            index.clear_cache();
        }
    }
    
    fn __repr__(&self) -> String {
        if self.memory_only {
            format!("UnifiedEngine(memory_only=True, terms={})", self.term_count())
        } else if self.lazy_load {
            format!("UnifiedEngine(lazy_load=True, terms={}, segments={})", 
                self.term_count(), self.segment_count())
        } else {
            format!("UnifiedEngine(terms={}, segments={})", 
                self.term_count(), self.segment_count())
        }
    }
}

// ==================== Internal Methods ====================

impl UnifiedEngine {
    fn add_text(&self, doc_id: u32, text: &str) {
        let terms = self.tokenize(text);
        
        if self.track_doc_terms {
            self.doc_terms.insert(doc_id, terms.clone());
        }
        
        for term in terms {
            self.buffer.entry(term.clone())
                .or_insert_with(FastBitmap::new)
                .add(doc_id);
            
            // Note: no longer writing to index simultaneously (double write)
            // Data only written to buffer, unified write to index on flush
            // This avoids O(n²) performance degradation
        }
    }
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();
        let text_lower = text.to_lowercase();
        
        // Chinese n-gram
        for cap in self.chinese_pattern.find_iter(&text_lower) {
            let chinese = cap.as_str();
            let chars: Vec<char> = chinese.chars().collect();
            
            for n in 2..=self.max_chinese_length.min(chars.len()) {
                for i in 0..=chars.len().saturating_sub(n) {
                    let term: String = chars[i..i+n].iter().collect();
                    if term.len() >= self.min_term_length {
                        terms.push(term);
                    }
                }
            }
        }
        
        // English tokenization - split by non-alphanumeric characters
        let english_only = self.chinese_pattern.replace_all(&text_lower, " ");
        for word in english_only.split(|c: char| !c.is_alphanumeric()) {
            let word = word.trim();
            if word.len() >= self.min_term_length {
                terms.push(word.to_string());
            }
        }
        
        terms.sort();
        terms.dedup();
        terms
    }
    
    fn search_internal(&self, query: &str) -> FastBitmap {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace()
            .filter(|w| w.len() >= self.min_term_length)
            .collect();
        
        if words.is_empty() {
            return FastBitmap::new();
        }
        
        let mut result: Option<FastBitmap> = None;
        
        for word in &words {
            let word_result = self.search_term(word);
            
            result = Some(match result {
                Some(mut r) => { r.and_inplace(&word_result); r }
                None => word_result,
            });
            
            if result.as_ref().map_or(true, |r| r.is_empty()) {
                return FastBitmap::new();
            }
        }
        
        let mut bitmap = result.unwrap_or_else(FastBitmap::new);
        
        // Exclude deleted documents (tombstone mechanism)
        if !self.deleted_docs.is_empty() {
            for entry in self.deleted_docs.iter() {
                bitmap.remove(*entry.key());
            }
        }
        
        // If track_doc_terms enabled, validate each result's validity
        // Filter out documents whose current terms don't contain query words (old version data)
        if self.track_doc_terms && !self.doc_terms.is_empty() {
            let query_terms: std::collections::HashSet<String> = self.tokenize(&query_lower)
                .into_iter()
                .collect();
            
            let mut valid_ids = Vec::new();
            for doc_id in bitmap.iter() {
                // If document has doc_terms record, check if it contains all query words
                if let Some(terms) = self.doc_terms.get(&doc_id) {
                    let doc_term_set: std::collections::HashSet<&String> = terms.iter().collect();
                    // Check if all query words are in document terms
                    let all_match = words.iter().all(|word| {
                        // For each query word, check if there's a matching document term
                        query_terms.iter().any(|qt| {
                            qt.contains(word) && doc_term_set.iter().any(|dt| dt.contains(word))
                        }) || doc_term_set.iter().any(|dt| dt.contains(word))
                    });
                    if all_match {
                        valid_ids.push(doc_id);
                    }
                } else {
                    // Documents without doc_terms record are kept directly
                    valid_ids.push(doc_id);
                }
            }
            bitmap = FastBitmap::from_iter(valid_ids);
        }
        
        bitmap
    }
    
    fn search_term(&self, term: &str) -> FastBitmap {
        let mut result = FastBitmap::new();
        
        // Query buffer
        if let Some(bitmap) = self.buffer.get(term) {
            result.or_inplace(&bitmap);
        }
        
        // Query index (exclude updated documents)
        if let Some(ref index) = self.index {
            if let Some(mut bitmap) = index.get(term) {
                // Remove updated documents from index results (their data should come from buffer)
                if !self.updated_docs.is_empty() {
                    for entry in self.updated_docs.iter() {
                        bitmap.remove(*entry.key());
                    }
                }
                result.or_inplace(&bitmap);
            }
        }
        
        // Chinese n-gram search
        if self.chinese_pattern.is_match(term) {
            let chars: Vec<char> = term.chars().collect();
            if chars.len() >= 2 {
                for n in 2..=self.max_chinese_length.min(chars.len()) {
                    for i in 0..=chars.len().saturating_sub(n) {
                        let ngram: String = chars[i..i+n].iter().collect();
                        
                        if let Some(bitmap) = self.buffer.get(&ngram) {
                            if result.is_empty() {
                                result = bitmap.clone();
                            } else {
                                result.and_inplace(&bitmap);
                            }
                        }
                        
                        if let Some(ref index) = self.index {
                            if let Some(mut bitmap) = index.get(&ngram) {
                                // Also exclude updated documents
                                if !self.updated_docs.is_empty() {
                                    for entry in self.updated_docs.iter() {
                                        bitmap.remove(*entry.key());
                                    }
                                }
                                if result.is_empty() {
                                    result = bitmap;
                                } else {
                                    result.and_inplace(&bitmap);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        result
    }
    
    fn fuzzy_search_internal(&self, query: &str) -> FastBitmap {
        let config = self.fuzzy_config.read();
        let threshold = config.threshold;
        let max_candidates = config.max_candidates;
        drop(config);
        
        // Collect all terms
        let mut all_terms: Vec<String> = self.buffer.iter()
            .map(|e| e.key().clone())
            .collect();
        
        if let Some(ref index) = self.index {
            all_terms.extend(index.all_terms());
        }
        
        // Find similar terms
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut result = FastBitmap::new();
        
        for word in words {
            let mut candidates: Vec<(String, f64)> = all_terms.iter()
                .filter_map(|t| {
                    let score = simd_utils::similarity_score(word, t);
                    if score >= threshold {
                        Some((t.clone(), score))
                    } else {
                        None
                    }
                })
                .collect();
            
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(max_candidates);
            
            let mut word_result = FastBitmap::new();
            for (term, _) in candidates {
                let term_bitmap = self.search_term(&term);
                word_result.or_inplace(&term_bitmap);
            }
            
            if result.is_empty() {
                result = word_result;
            } else {
                result.and_inplace(&word_result);
            }
        }
        
        result
    }
}

/// Create unified search engine
/// 
/// Args:
///     index_file: Index file path, empty string for memory-only mode
///     max_chinese_length: Maximum Chinese n-gram length (default 4)
///     min_term_length: Minimum term length (default 2)
///     fuzzy_threshold: Fuzzy search similarity threshold (default 0.7)
///     fuzzy_max_distance: Fuzzy search maximum edit distance (default 2)
///     track_doc_terms: Whether to track document terms (default False)
///     drop_if_exists: Whether to delete existing index file (default False)
///     lazy_load: Whether to enable lazy load mode (default True)
///     cache_size: LRU cache size in lazy load mode (default 10000)
/// 
/// Returns:
///     UnifiedEngine: Search engine instance
/// 
/// Example:
///     # Default mode (lazy load, recommended)
///     engine = create_engine("index.nfts")
///     
///     # Full load mode (suitable for small indexes)
///     engine = create_engine("index.nfts", lazy_load=False)
#[pyfunction]
#[pyo3(signature = (
    index_file="".to_string(),
    max_chinese_length=4,
    min_term_length=2,
    fuzzy_threshold=0.7,
    fuzzy_max_distance=2,
    track_doc_terms=false,
    drop_if_exists=false,
    lazy_load=false,
    cache_size=10000
))]
pub fn create_engine(
    index_file: String,
    max_chinese_length: usize,
    min_term_length: usize,
    fuzzy_threshold: f64,
    fuzzy_max_distance: usize,
    track_doc_terms: bool,
    drop_if_exists: bool,
    lazy_load: bool,
    cache_size: usize,
) -> PyResult<UnifiedEngine> {
    UnifiedEngine::new(
        index_file,
        max_chinese_length,
        min_term_length,
        fuzzy_threshold,
        fuzzy_max_distance,
        track_doc_terms,
        drop_if_exists,
        lazy_load,
        cache_size,
    )
}
