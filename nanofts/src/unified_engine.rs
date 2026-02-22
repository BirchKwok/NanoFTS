//! Unified Search Engine - Single-File LSM Implementation
//!
//! Replacement for PagedEngine, LsmEngine, LsmSingleEngine
//! Supports: memory-only mode, single-file persistence, fuzzy search, document delete/update
//!
//! # Rust API Example
//!
//! ```rust
//! use nanofts::{UnifiedEngine, EngineConfig};
//!
//! // Create in-memory engine
//! let engine = UnifiedEngine::new(EngineConfig::default()).unwrap();
//!
//! // Add documents
//! let mut fields = std::collections::HashMap::new();
//! fields.insert("title".to_string(), "Hello World".to_string());
//! engine.add_document(1, fields).unwrap();
//!
//! // Search
//! let result = engine.search("hello").unwrap();
//! println!("Found {} documents", result.total_hits());
//! ```

use crate::bitmap::FastBitmap;
use crate::lsm_single::LsmSingleIndex;
use crate::simd_utils;
use dashmap::DashMap;
use fork_union::spawn;
use parking_lot::{RwLock, Mutex};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Engine error type
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Index error: {0}")]
    IndexError(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

#[cfg(feature = "python")]
impl From<EngineError> for PyErr {
    fn from(err: EngineError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// Result type for engine operations
pub type EngineResult<T> = Result<T, EngineError>;

/// Search result handle
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[derive(Clone)]
pub struct ResultHandle {
    bitmap: Arc<FastBitmap>,
    query: String,
    elapsed_ns: u64,
    fuzzy_used: bool,
}

// Pure Rust API
impl ResultHandle {
    /// Create a new result handle
    pub fn new(bitmap: FastBitmap, query: String, elapsed_ns: u64, fuzzy_used: bool) -> Self {
        Self {
            bitmap: Arc::new(bitmap),
            query,
            elapsed_ns,
            fuzzy_used,
        }
    }
    
    /// Get total number of hits
    pub fn total_hits(&self) -> u64 {
        self.bitmap.len()
    }
    
    /// Get elapsed time in nanoseconds
    pub fn get_elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }
    
    /// Check if fuzzy search was used
    pub fn is_fuzzy_used(&self) -> bool {
        self.fuzzy_used
    }
    
    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000.0
    }
    
    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000.0
    }
    
    /// Check if result contains document ID
    pub fn contains(&self, doc_id: u32) -> bool {
        self.bitmap.contains(doc_id)
    }
    
    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }
    
    /// Get top N document IDs
    pub fn top(&self, n: usize) -> Vec<u32> {
        self.bitmap.iter().take(n).collect()
    }
    
    /// Convert to vector of document IDs
    pub fn to_list(&self) -> Vec<u32> {
        self.bitmap.to_vec()
    }
    
    /// Get page of results
    pub fn page(&self, offset: usize, limit: usize) -> Vec<u32> {
        self.bitmap.iter().skip(offset).take(limit).collect()
    }
    
    /// Intersection with another result
    pub fn intersect(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.and(&other.bitmap)),
            query: format!("({}) AND ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used || other.fuzzy_used,
        }
    }
    
    /// Union with another result
    pub fn union(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.or(&other.bitmap)),
            query: format!("({}) OR ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used || other.fuzzy_used,
        }
    }
    
    /// Difference with another result
    pub fn difference(&self, other: &ResultHandle) -> ResultHandle {
        let start = std::time::Instant::now();
        ResultHandle {
            bitmap: Arc::new(self.bitmap.andnot(&other.bitmap)),
            query: format!("({}) NOT ({})", self.query, other.query),
            elapsed_ns: start.elapsed().as_nanos() as u64,
            fuzzy_used: self.fuzzy_used,
        }
    }
    
    /// Get query string
    pub fn query(&self) -> &str {
        &self.query
    }
    
    /// Get length (number of hits)
    pub fn len(&self) -> usize {
        self.bitmap.len() as usize
    }
    
    /// Get iterator over document IDs
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bitmap.iter()
    }
}

impl std::fmt::Display for ResultHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ResultHandle(hits={}, query='{}', elapsed={:.3}ms)",
            self.bitmap.len(), self.query, self.elapsed_ns as f64 / 1_000_000.0
        )
    }
}

// Python-specific methods
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl ResultHandle {
    #[getter(total_hits)]
    fn total_hits_py(&self) -> u64 {
        self.total_hits()
    }
    
    #[getter]
    fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }
    
    #[getter]
    fn fuzzy_used(&self) -> bool {
        self.fuzzy_used
    }
    
    #[pyo3(name = "elapsed_ms")]
    fn elapsed_ms_py(&self) -> f64 {
        self.elapsed_ms()
    }
    
    #[pyo3(name = "elapsed_us")]
    fn elapsed_us_py(&self) -> f64 {
        self.elapsed_us()
    }
    
    #[pyo3(name = "contains")]
    fn contains_py(&self, doc_id: u32) -> bool {
        self.contains(doc_id)
    }
    
    #[pyo3(name = "is_empty")]
    fn is_empty_py(&self) -> bool {
        self.is_empty()
    }
    
    #[pyo3(signature = (n=100))]
    #[pyo3(name = "top")]
    fn top_py(&self, n: usize) -> Vec<u32> {
        self.top(n)
    }
    
    #[pyo3(name = "to_list")]
    fn to_list_py(&self) -> Vec<u32> {
        self.to_list()
    }
    
    fn to_numpy<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<u32>>> {
        let ids: Vec<u32> = self.bitmap.to_vec();
        Ok(numpy::PyArray1::from_vec_bound(py, ids))
    }
    
    #[pyo3(name = "page")]
    fn page_py(&self, offset: usize, limit: usize) -> Vec<u32> {
        self.page(offset, limit)
    }
    
    #[pyo3(name = "intersect")]
    fn intersect_py(&self, other: &ResultHandle) -> ResultHandle {
        self.intersect(other)
    }
    
    #[pyo3(name = "union")]
    fn union_py(&self, other: &ResultHandle) -> ResultHandle {
        self.union(other)
    }
    
    #[pyo3(name = "difference")]
    fn difference_py(&self, other: &ResultHandle) -> ResultHandle {
        self.difference(other)
    }
    
    fn __len__(&self) -> usize {
        self.len()
    }
    
    fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Fuzzy search configuration
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
pub struct FuzzyConfig {
    pub threshold: f64,
    pub max_distance: usize,
    pub max_candidates: usize,
}

impl FuzzyConfig {
    /// Create new fuzzy config with default values
    pub fn new(threshold: f64, max_distance: usize, max_candidates: usize) -> Self {
        Self { threshold, max_distance, max_candidates }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl FuzzyConfig {
    #[new]
    #[pyo3(signature = (threshold=0.7, max_distance=2, max_candidates=20))]
    fn py_new(threshold: f64, max_distance: usize, max_candidates: usize) -> Self {
        Self::new(threshold, max_distance, max_candidates)
    }
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self { threshold: 0.7, max_distance: 2, max_candidates: 20 }
    }
}

/// Engine configuration
#[derive(Clone)]
pub struct EngineConfig {
    /// Index file path, empty for memory-only mode
    pub index_file: String,
    /// Maximum Chinese n-gram length
    pub max_chinese_length: usize,
    /// Minimum term length
    pub min_term_length: usize,
    /// Fuzzy search similarity threshold
    pub fuzzy_threshold: f64,
    /// Fuzzy search maximum edit distance
    pub fuzzy_max_distance: usize,
    /// Whether to track document terms (for efficient deletion)
    pub track_doc_terms: bool,
    /// Whether to delete existing index file
    pub drop_if_exists: bool,
    /// Whether to enable lazy load mode
    pub lazy_load: bool,
    /// LRU cache size in lazy load mode
    pub cache_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            index_file: String::new(),
            max_chinese_length: 4,
            min_term_length: 2,
            fuzzy_threshold: 0.7,
            fuzzy_max_distance: 2,
            track_doc_terms: false,
            drop_if_exists: false,
            lazy_load: false,
            cache_size: 10000,
        }
    }
}

impl EngineConfig {
    /// Create config for memory-only mode
    pub fn memory_only() -> Self {
        Self::default()
    }
    
    /// Create config for persistent mode
    pub fn persistent<S: Into<String>>(index_file: S) -> Self {
        Self {
            index_file: index_file.into(),
            ..Default::default()
        }
    }
    
    /// Enable lazy load mode
    pub fn with_lazy_load(mut self, enabled: bool) -> Self {
        self.lazy_load = enabled;
        self
    }
    
    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
    
    /// Enable document term tracking
    pub fn with_track_doc_terms(mut self, enabled: bool) -> Self {
        self.track_doc_terms = enabled;
        self
    }
    
    /// Set fuzzy search threshold
    pub fn with_fuzzy_threshold(mut self, threshold: f64) -> Self {
        self.fuzzy_threshold = threshold;
        self
    }
    
    /// Drop existing index if exists
    pub fn with_drop_if_exists(mut self, drop: bool) -> Self {
        self.drop_if_exists = drop;
        self
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
///
/// # Rust API Example
///
/// ```rust,no_run
/// use nanofts::{UnifiedEngine, EngineConfig};
/// use std::collections::HashMap;
///
/// let config = EngineConfig::persistent("index.nfts")
///     .with_lazy_load(true)
///     .with_cache_size(10000);
///
/// let engine = UnifiedEngine::new(config).unwrap();
///
/// // Add document
/// let mut fields = HashMap::new();
/// fields.insert("title".to_string(), "Hello World".to_string());
/// engine.add_document(1, fields).unwrap();
///
/// // Search
/// let result = engine.search("hello").unwrap();
/// println!("Found {} results", result.total_hits());
/// ```
#[cfg_attr(feature = "python", pyo3::pyclass(subclass))]
pub struct UnifiedEngine {
    // Dictionary encoding
    dict: Arc<dashmap::DashMap<String, u32>>,
    id_to_str: Arc<dashmap::DashMap<u32, String>>,
    next_term_id: std::sync::atomic::AtomicU32,
    
    // Storage layer
    index: Option<Arc<LsmSingleIndex>>,
    // Memory buffer (memory-only mode or write buffer)
    buffer: DashMap<u32, FastBitmap>,
    // Document-term mapping (for efficient deletion)
    doc_terms: DashMap<u32, Vec<u32>>,
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
    // Background flush handle (for flush_async)
    flush_handle: Mutex<Option<std::thread::JoinHandle<Result<usize, String>>>>,
}

// Pure Rust API implementation
impl UnifiedEngine {
    /// Get or insert a term into the global dictionary, returning its ID
    #[inline]
    fn get_term_id(&self, term: &str) -> u32 {
        if let Some(id_ref) = self.dict.get(term) {
            return *id_ref;
        }
        
        let new_id = self.next_term_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let term_owned = term.to_string();
        
        self.dict.insert(term_owned.clone(), new_id);
        self.id_to_str.insert(new_id, term_owned);
        
        new_id
    }
    
    /// Get the term string for a given ID
    #[inline]
    fn get_term_str(&self, id: u32) -> Option<String> {
        self.id_to_str.get(&id).map(|s| s.clone())
    }

    /// Build a reverse mapping id→str from the global dict (used at flush time)
    fn build_id_to_str_map(&self) -> FxHashMap<u32, String> {
        self.dict.iter().map(|e| (*e.value(), e.key().clone())).collect()
    }

    /// Create unified search engine with configuration
    pub fn new(config: EngineConfig) -> EngineResult<Self> {
        let memory_only = config.index_file.is_empty();
        
        let index = if memory_only {
            None
        } else {
            let path = std::path::PathBuf::from(&config.index_file);
            
            let idx = if path.exists() && config.drop_if_exists {
                std::fs::remove_file(&path)?;
                LsmSingleIndex::create_full_options(&path, true, config.lazy_load, config.cache_size)
            } else if path.exists() {
                LsmSingleIndex::open_full_options(&path, true, config.lazy_load, config.cache_size)
            } else {
                LsmSingleIndex::create_full_options(&path, true, config.lazy_load, config.cache_size)
            }.map_err(|e| EngineError::IndexError(e.to_string()))?;
            
            Some(Arc::new(idx))
        };
        
        Ok(Self {
            dict: Arc::new(dashmap::DashMap::with_shard_amount(64)),
            id_to_str: Arc::new(dashmap::DashMap::with_shard_amount(64)),
            next_term_id: std::sync::atomic::AtomicU32::new(1),
            index,
            buffer: DashMap::with_shard_amount(64),
            doc_terms: DashMap::new(),
            track_doc_terms: config.track_doc_terms,
            deleted_docs: DashMap::new(),
            updated_docs: DashMap::new(),
            max_chinese_length: config.max_chinese_length,
            min_term_length: config.min_term_length,
            fuzzy_config: RwLock::new(FuzzyConfig {
                threshold: config.fuzzy_threshold,
                max_distance: config.fuzzy_max_distance,
                max_candidates: 20,
            }),
            result_cache: DashMap::new(),
            cache_enabled: true,
            stats: EngineStats::default(),
            memory_only,
            lazy_load: config.lazy_load,
            preloaded: std::sync::atomic::AtomicBool::new(false),
            chinese_pattern: regex::Regex::new(r"[\u4e00-\u9fff]+").unwrap(),
            compact_lock: RwLock::new(()),
            flush_handle: Mutex::new(None),
        })
    }
    
    /// Create engine from raw parameters (used by Python bindings)
    pub fn new_with_params(
        index_file: String,
        max_chinese_length: usize,
        min_term_length: usize,
        fuzzy_threshold: f64,
        fuzzy_max_distance: usize,
        track_doc_terms: bool,
        drop_if_exists: bool,
        lazy_load: bool,
        cache_size: usize,
    ) -> EngineResult<Self> {
        Self::new(EngineConfig {
            index_file,
            max_chinese_length,
            min_term_length,
            fuzzy_threshold,
            fuzzy_max_distance,
            track_doc_terms,
            drop_if_exists,
            lazy_load,
            cache_size,
        })
    }
    
    // ==================== Document Operations ====================
    
    /// Add single document
    pub fn add_document(&self, doc_id: u32, fields: HashMap<String, String>) -> EngineResult<()> {
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
    
    /// Batch add documents - optimized parallel version using ForkUnion
    pub fn add_documents(&self, docs: Vec<(u32, HashMap<String, String>)>) -> EngineResult<usize> {
        let count = docs.len();
        if count == 0 {
            return Ok(0);
        }
        
        // Clear deleted/updated markers for all docs
        for (doc_id, _) in &docs {
            self.deleted_docs.remove(doc_id);
            self.updated_docs.remove(doc_id);
        }
        
        // Use optimized batch processing
        self.add_documents_batch_parallel(&docs);
        
        self.result_cache.clear();
        Ok(count)
    }
    
    /// Batch add documents with columnar data - optimized for Arrow/DataFrame-like input
    /// 
    /// This method accepts columnar data format which is more efficient when data comes from
    /// Arrow, pandas DataFrame, or other columnar sources. It avoids the overhead of 
    /// constructing HashMap for each document.
    ///
    /// # Arguments
    /// * `doc_ids` - Vector of document IDs
    /// * `columns` - Vector of (field_name, field_values) pairs, where field_values is a Vec<String>
    ///               with the same length as doc_ids
    ///
    /// # Example
    /// ```rust,ignore
    /// // From pandas: df['id'].tolist(), [('title', df['title'].tolist()), ('content', df['content'].tolist())]
    /// engine.add_documents_columnar(
    ///     vec![1, 2, 3],
    ///     vec![
    ///         ("title".to_string(), vec!["Doc1".to_string(), "Doc2".to_string(), "Doc3".to_string()]),
    ///         ("content".to_string(), vec!["Content1".to_string(), "Content2".to_string(), "Content3".to_string()]),
    ///     ]
    /// )?;
    /// ```
    pub fn add_documents_columnar(
        &self, 
        doc_ids: Vec<u32>, 
        columns: Vec<(String, Vec<String>)>
    ) -> EngineResult<usize> {
        let num_docs = doc_ids.len();
        if num_docs == 0 {
            return Ok(0);
        }
        
        // Validate column lengths
        for (field_name, values) in &columns {
            if values.len() != num_docs {
                return Err(EngineError::InvalidArgument(format!(
                    "Column '{}' has {} values, expected {} (same as doc_ids)",
                    field_name, values.len(), num_docs
                )));
            }
        }
        
        // Clear deleted/updated markers for all docs
        for doc_id in &doc_ids {
            self.deleted_docs.remove(doc_id);
            self.updated_docs.remove(doc_id);
        }
        
        // Use optimized columnar batch processing
        self.add_documents_columnar_parallel(&doc_ids, &columns);
        
        self.result_cache.clear();
        Ok(num_docs)
    }
    
    /// Batch add documents with single text column - simplest columnar format
    /// 
    /// This is the fastest path when you have pre-concatenated text for each document.
    ///
    /// # Arguments
    /// * `doc_ids` - Vector of document IDs
    /// * `texts` - Vector of text content, same length as doc_ids
    pub fn add_documents_texts(&self, doc_ids: Vec<u32>, texts: Vec<String>) -> EngineResult<usize> {
        let num_docs = doc_ids.len();
        if num_docs == 0 {
            return Ok(0);
        }
        
        if texts.len() != num_docs {
            return Err(EngineError::InvalidArgument(format!(
                "texts has {} values, expected {} (same as doc_ids)",
                texts.len(), num_docs
            )));
        }
        
        // Clear deleted/updated markers for all docs
        for doc_id in &doc_ids {
            self.deleted_docs.remove(doc_id);
            self.updated_docs.remove(doc_id);
        }
        
        // Use optimized single-column batch processing
        self.add_documents_texts_parallel(&doc_ids, &texts);
        
        self.result_cache.clear();
        Ok(num_docs)
    }
    
    /// Add documents from Arrow-style columnar data with minimal copying
    /// 
    /// This method accepts pre-extracted string slices from Arrow StringArray,
    /// avoiding the need to clone String data from Arrow buffers.
    /// 
    /// # Arguments
    /// * `doc_ids` - Document IDs as u32 slice
    /// * `columns` - Column data as vector of (field_name, string_slices) pairs
    /// 
    /// # Performance
    /// This is ~10-20% faster than add_documents_columnar when data comes from Arrow
    /// because it avoids allocating Vec<String> and copying string data.
    /// 
    /// # Example
    /// ```rust,ignore
    /// // Extract string slices from Arrow StringArray
    /// let title_slices: Vec<&str> = title_array.iter().map(|s| s.unwrap_or("")).collect();
    /// let content_slices: Vec<&str> = content_array.iter().map(|s| s.unwrap_or("")).collect();
    /// 
    /// engine.add_documents_arrow_str(
    ///     &doc_ids,
    ///     vec![
    ///         ("title".to_string(), title_slices),
    ///         ("content".to_string(), content_slices),
    ///     ]
    /// )?;
    /// ```
    pub fn add_documents_arrow_str<'a>(
        &self,
        doc_ids: &[u32],
        columns: Vec<(String, Vec<&'a str>)>,
    ) -> EngineResult<usize> {
        let num_docs = doc_ids.len();
        if num_docs == 0 {
            return Ok(0);
        }
        
        // Validate column lengths
        for (field_name, values) in &columns {
            if values.len() != num_docs {
                return Err(EngineError::InvalidArgument(format!(
                    "Column '{}' has {} values, expected {} (same as doc_ids)",
                    field_name, values.len(), num_docs
                )));
            }
        }
        
        // Clear deleted/updated markers
        for doc_id in doc_ids {
            self.deleted_docs.remove(doc_id);
            self.updated_docs.remove(doc_id);
        }
        
        // Use optimized zero-copy batch processing
        self.add_documents_arrow_parallel(doc_ids, &columns);
        
        self.result_cache.clear();
        Ok(num_docs)
    }
    
    /// Add documents from Arrow single text column with zero-copy
    /// 
    /// Fastest path for Arrow data - pre-concatenated text as string slices.
    /// 
    /// # Arguments
    /// * `doc_ids` - Document IDs as u32 slice
    /// * `texts` - Text content as string slices from Arrow StringArray
    pub fn add_documents_arrow_texts<'a>(
        &self,
        doc_ids: &[u32],
        texts: &[&'a str],
    ) -> EngineResult<usize> {
        let num_docs = doc_ids.len();
        if num_docs == 0 {
            return Ok(0);
        }
        
        if texts.len() != num_docs {
            return Err(EngineError::InvalidArgument(format!(
                "texts has {} values, expected {} (same as doc_ids)",
                texts.len(), num_docs
            )));
        }
        
        // Clear deleted/updated markers
        for doc_id in doc_ids {
            self.deleted_docs.remove(doc_id);
            self.updated_docs.remove(doc_id);
        }
        
        // Use optimized zero-copy single-column processing
        self.add_documents_arrow_texts_parallel(doc_ids, texts);
        
        self.result_cache.clear();
        Ok(num_docs)
    }
    
    /// Update document
    pub fn update_document(&self, doc_id: u32, fields: HashMap<String, String>) -> EngineResult<()> {
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
    pub fn remove_document(&self, doc_id: u32) -> EngineResult<()> {
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
    pub fn remove_documents(&self, doc_ids: Vec<u32>) -> EngineResult<()> {
        for doc_id in doc_ids {
            self.remove_document(doc_id)?;
        }
        Ok(())
    }
    
    // ==================== Search Operations ====================
    
    /// Search
    pub fn search(&self, query: &str) -> EngineResult<ResultHandle> {
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
    pub fn fuzzy_search(&self, query: &str, min_results: usize) -> EngineResult<ResultHandle> {
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
    
    /// Batch search - optimized parallel version using ForkUnion
    pub fn search_batch(&self, queries: Vec<String>) -> EngineResult<Vec<ResultHandle>> {
        let num_queries = queries.len();
        if num_queries == 0 {
            return Ok(Vec::new());
        }
        
        // Pre-allocate results with padding to avoid false sharing
        let results: Vec<Mutex<Option<ResultHandle>>> = 
            (0..num_queries)
                .map(|_| Mutex::new(None))
                .collect();
        
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .max(1);
        
        let mut pool = spawn(num_threads);
        
        pool.for_n(num_queries, |prong| {
            let idx = prong.task_index;
            let q = &queries[idx];
            let start = std::time::Instant::now();
            let bitmap = Arc::new(self.search_internal(q));
            let handle = ResultHandle {
                bitmap,
                query: q.clone(),
                elapsed_ns: start.elapsed().as_nanos() as u64,
                fuzzy_used: false,
            };
            *results[idx].lock() = Some(handle);
        });
        
        // Collect results in order
        let final_results: Vec<ResultHandle> = results
            .into_iter()
            .map(|r| r.into_inner().unwrap())
            .collect();
        
        Ok(final_results)
    }
    
    /// AND search
    pub fn search_and(&self, queries: Vec<String>) -> EngineResult<ResultHandle> {
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
    pub fn search_or(&self, queries: Vec<String>) -> EngineResult<ResultHandle> {
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
    pub fn filter_by_ids(&self, result: &ResultHandle, allowed_ids: Vec<u32>) -> EngineResult<ResultHandle> {
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
    pub fn exclude_ids(&self, result: &ResultHandle, excluded_ids: Vec<u32>) -> EngineResult<ResultHandle> {
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
    
    /// Set fuzzy search configuration
    pub fn set_fuzzy_config(&self, threshold: f64, max_distance: usize, max_candidates: usize) {
        let mut config = self.fuzzy_config.write();
        config.threshold = threshold;
        config.max_distance = max_distance;
        config.max_candidates = max_candidates;
    }
    
    /// Get fuzzy search configuration
    pub fn get_fuzzy_config(&self) -> HashMap<String, f64> {
        let config = self.fuzzy_config.read();
        let mut map = HashMap::new();
        map.insert("threshold".to_string(), config.threshold);
        map.insert("max_distance".to_string(), config.max_distance as f64);
        map.insert("max_candidates".to_string(), config.max_candidates as f64);
        map
    }
    
    // ==================== Persistence Operations ====================
    
    /// Flush to disk
    pub fn flush(&self) -> EngineResult<usize> {
        if self.memory_only {
            return Ok(0);
        }
        
        // Get read lock to prevent flush during compact
        let _lock = self.compact_lock.read();
        
        if let Some(ref index) = self.index {
            let mut entries: Vec<(String, FastBitmap)> = Vec::new();
            
            // Drain buffer; look up term strings directly from id_to_str (avoids
            // the O(dict) build_id_to_str_map copy).
            let keys: Vec<u32> = self.buffer.iter().map(|e| *e.key()).collect();
            for key in keys {
                if let Some((term_id, bitmap)) = self.buffer.remove(&key) {
                    if let Some(term_str) = self.id_to_str.get(&term_id).map(|r| r.value().clone()) {
                        entries.push((term_str, bitmap));
                    }
                }
            }
            
            let count = entries.len();
            
            if !entries.is_empty() {
                // Clean updated_docs for all flushed documents.
                for (_, bitmap) in &entries {
                    for doc_id in bitmap.iter() {
                        self.updated_docs.remove(&doc_id);
                    }
                }
                index.upsert_batch(entries);
                index.flush()
                    .map_err(|e| EngineError::IndexError(e.to_string()))?;
            }
            
            Ok(count)
        } else {
            Ok(0)
        }
    }
    
    /// Flush to disk asynchronously — returns immediately, disk I/O runs in a background thread.
    ///
    /// Data is still searchable after this call returns (it remains in `LsmSingleIndex`'s
    /// in-memory buffer until the background thread persists it). Call [`wait_flush`] to
    /// block until the background write completes.
    ///
    /// **Use case**: bulk index builds where you want to start searching immediately without
    /// waiting for fsync. Not intended as a replacement for [`flush`] in write-heavy
    /// incremental workloads.
    pub fn flush_async(&self) -> EngineResult<()> {
        if self.memory_only {
            return Ok(());
        }

        let _lock = self.compact_lock.read();

        if let Some(ref index) = self.index {
            // --- Sync part (fast): drain UnifiedEngine buffer ---
            // Look up term strings directly in id_to_str (no full-dict copy).
            let keys: Vec<u32> = self.buffer.iter().map(|e| *e.key()).collect();
            let mut entries: Vec<(String, FastBitmap)> = Vec::new();

            for key in keys {
                if let Some((term_id, bitmap)) = self.buffer.remove(&key) {
                    if let Some(term_str) = self.id_to_str.get(&term_id).map(|r| r.value().clone()) {
                        entries.push((term_str, bitmap));
                    }
                }
            }

            self.result_cache.clear();

            // Pre-clean updated_docs for all flushed documents.
            for (_, bitmap) in &entries {
                for doc_id in bitmap.iter() {
                    self.updated_docs.remove(&doc_id);
                }
            }

            let count = entries.len();

            if !entries.is_empty() {
                // --- Fast sync part: make data searchable immediately, zero disk I/O ---
                // merge_into_data writes directly to LsmSingleIndex.data (full-load mode),
                // bypassing WAL and the buffer-threshold auto-flush so this returns in
                // microseconds regardless of how many entries there are.
                index.merge_into_data(&entries);

                // --- Async part: all disk I/O (WAL-less buffer write + fsync) in background ---
                let index_clone = Arc::clone(index);
                let handle = std::thread::spawn(move || {
                    // Put entries into LsmSingleIndex.buffer without WAL and without
                    // triggering maybe_flush; the explicit flush() below handles persistence.
                    index_clone.enqueue_for_flush(entries);
                    index_clone.flush()
                        .map_err(|e| e.to_string())?;
                    Ok::<usize, String>(count)
                });

                *self.flush_handle.lock() = Some(handle);
            }
        }

        Ok(())
    }

    /// Wait for a previously started [`flush_async`] to complete.
    ///
    /// Returns the number of terms flushed, or an error if the background thread
    /// failed or panicked. Returns `Ok(0)` if no background flush is pending.
    pub fn wait_flush(&self) -> EngineResult<usize> {
        let handle = self.flush_handle.lock().take();
        match handle {
            Some(h) => {
                let result = h
                    .join()
                    .map_err(|_| EngineError::IndexError("Background flush thread panicked".to_string()))?
                    .map_err(|e| EngineError::IndexError(e));
                // Invalidate any cached results that were computed while disk write was in flight
                self.result_cache.clear();
                result
            }
            None => Ok(0),
        }
    }

    /// Save (same as flush)
    pub fn save(&self) -> EngineResult<usize> {
        self.flush()
    }
    
    /// Load (no-op, data is loaded on open)
    pub fn load(&self) -> EngineResult<()> {
        self.result_cache.clear();
        Ok(())
    }
    
    /// Preload
    pub fn preload(&self) -> EngineResult<usize> {
        if self.memory_only || self.preloaded.load(Ordering::Relaxed) {
            return Ok(0);
        }
        
        if let Some(ref index) = self.index {
            self.preloaded.store(true, Ordering::Relaxed);
            Ok(index.term_count())
        } else {
            Ok(0)
        }
    }
    
    /// Compact (also applies deletions)
    pub fn compact(&self) -> EngineResult<()> {
        if let Some(ref index) = self.index {
            let _lock = self.compact_lock.write();
            
            let mut pending_entries: Vec<(String, FastBitmap)> = Vec::new();
            let mut flushed_doc_ids = std::collections::HashSet::new();
            {
                let id_to_str_map = self.build_id_to_str_map();
                let keys: Vec<u32> = self.buffer.iter().map(|e| *e.key()).collect();
                for key in keys {
                    if let Some((term_id, bitmap)) = self.buffer.remove(&key) {
                        if let Some(term_str) = id_to_str_map.get(&term_id).cloned() {
                            for doc_id in bitmap.iter() {
                                flushed_doc_ids.insert(doc_id);
                            }
                            pending_entries.push((term_str, bitmap));
                        }
                    }
                }
            }
            
            let deleted: Vec<u32> = self.deleted_docs.iter()
                .map(|e| *e.key())
                .chain(self.updated_docs.iter().map(|e| *e.key()))
                .collect();
            
            index.compact_with_deletions(&deleted)
                .map_err(|e| EngineError::IndexError(e.to_string()))?;
            
            if !pending_entries.is_empty() {
                index.upsert_batch(pending_entries);
                index.flush()
                    .map_err(|e| EngineError::IndexError(e.to_string()))?;
            }
            
            for doc_id in flushed_doc_ids {
                self.updated_docs.remove(&doc_id);
            }
            
            {
                let mut flushed_doc_ids = std::collections::HashSet::new();
                let mut entries: Vec<(String, FastBitmap)> = Vec::new();
                
                let id_to_str_map2 = self.build_id_to_str_map();
                let keys: Vec<u32> = self.buffer.iter().map(|e| *e.key()).collect();
                for key in keys {
                    if let Some((term_id, bitmap)) = self.buffer.remove(&key) {
                        if let Some(term_str) = id_to_str_map2.get(&term_id).cloned() {
                            for doc_id in bitmap.iter() {
                                flushed_doc_ids.insert(doc_id);
                            }
                            entries.push((term_str, bitmap));
                        }
                    }
                }
                
                if !entries.is_empty() {
                    index.upsert_batch(entries);
                    index.flush()
                        .map_err(|e| EngineError::IndexError(e.to_string()))?;
                    
                    for doc_id in flushed_doc_ids {
                        self.updated_docs.remove(&doc_id);
                    }
                }
            }
            
            self.deleted_docs.clear();
            self.updated_docs.clear();
        }
        Ok(())
    }
    
    // ==================== Cache Operations ====================
    
    /// Clear result cache
    pub fn clear_cache(&self) {
        self.result_cache.clear();
    }
    
    /// Clear doc terms
    pub fn clear_doc_terms(&self) {
        self.doc_terms.clear();
    }
    
    // ==================== Statistics Operations ====================
    
    /// Check if memory only mode
    pub fn is_memory_only(&self) -> bool {
        self.memory_only
    }
    
    /// Get term count
    pub fn term_count(&self) -> usize {
        let buffer_count = self.buffer.len();
        let index_count = self.index.as_ref().map_or(0, |i| i.term_count());
        buffer_count + index_count
    }
    
    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    
    /// Get doc terms size
    pub fn doc_terms_size(&self) -> usize {
        self.doc_terms.len()
    }
    
    /// Get page count
    pub fn page_count(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.segment_count())
    }
    
    /// Get segment count
    pub fn segment_count(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.segment_count())
    }
    
    /// Get memtable size
    pub fn memtable_size(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.memtable_size())
    }
    
    /// Get statistics
    pub fn stats(&self) -> HashMap<String, f64> {
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
        
        if let Some(ref index) = self.index {
            map.insert("wal_enabled".to_string(), if index.is_wal_enabled() { 1.0 } else { 0.0 });
            map.insert("wal_size".to_string(), index.wal_size() as f64);
            map.insert("wal_pending_batches".to_string(), index.wal_pending_batches() as f64);
            
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
    pub fn deleted_count(&self) -> usize {
        self.deleted_docs.len()
    }
    
    /// Whether lazy load is enabled
    pub fn is_lazy_load(&self) -> bool {
        self.lazy_load
    }
    
    /// Warmup cache
    pub fn warmup_terms(&self, terms: Vec<String>) -> usize {
        if let Some(ref index) = self.index {
            index.warmup(&terms)
        } else {
            0
        }
    }
    
    /// Clear LRU cache
    pub fn clear_lru_cache(&self) {
        if let Some(ref index) = self.index {
            index.clear_cache();
        }
    }
}

impl std::fmt::Display for UnifiedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.memory_only {
            write!(f, "UnifiedEngine(memory_only=true, terms={})", self.term_count())
        } else if self.lazy_load {
            write!(f, "UnifiedEngine(lazy_load=true, terms={}, segments={})", 
                self.term_count(), self.segment_count())
        } else {
            write!(f, "UnifiedEngine(terms={}, segments={})", 
                self.term_count(), self.segment_count())
        }
    }
}

// Python bindings
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl UnifiedEngine {
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
    fn py_new(
        index_file: String,
        max_chinese_length: usize,
        min_term_length: usize,
        fuzzy_threshold: f64,
        fuzzy_max_distance: usize,
        track_doc_terms: bool,
        drop_if_exists: bool,
        lazy_load: bool,
        cache_size: usize,
    ) -> pyo3::PyResult<Self> {
        Self::new_with_params(
            index_file,
            max_chinese_length,
            min_term_length,
            fuzzy_threshold,
            fuzzy_max_distance,
            track_doc_terms,
            drop_if_exists,
            lazy_load,
            cache_size,
        ).map_err(Into::into)
    }
    
    #[pyo3(name = "add_document")]
    fn add_document_py(&self, doc_id: u32, fields: HashMap<String, String>) -> pyo3::PyResult<()> {
        self.add_document(doc_id, fields).map_err(Into::into)
    }
    
    #[pyo3(name = "add_documents")]
    fn add_documents_py(&self, docs: Vec<(u32, HashMap<String, String>)>) -> pyo3::PyResult<usize> {
        self.add_documents(docs).map_err(Into::into)
    }
    
    /// Add documents using columnar data format - optimized for Arrow/DataFrame input
    /// 
    /// This method is more efficient when data comes from pandas DataFrame or PyArrow.
    /// It avoids the overhead of constructing Python dicts for each document.
    ///
    /// # Arguments
    /// * `doc_ids` - List of document IDs (can be numpy array or Python list)
    /// * `columns` - List of (field_name, field_values) tuples, where field_values 
    ///               is a list of strings with the same length as doc_ids
    ///
    /// # Example
    /// ```python
    /// import pandas as pd
    /// df = pd.DataFrame({'id': [1, 2, 3], 'title': ['A', 'B', 'C'], 'content': ['X', 'Y', 'Z']})
    /// 
    /// # Columnar format - faster for large datasets
    /// engine.add_documents_columnar(
    ///     df['id'].tolist(),
    ///     [('title', df['title'].tolist()), ('content', df['content'].tolist())]
    /// )
    /// ```
    #[pyo3(name = "add_documents_columnar")]
    fn add_documents_columnar_py(
        &self, 
        doc_ids: Vec<u32>, 
        columns: Vec<(String, Vec<String>)>
    ) -> pyo3::PyResult<usize> {
        self.add_documents_columnar(doc_ids, columns).map_err(Into::into)
    }
    
    /// Add documents using single text column - fastest path for pre-concatenated text
    ///
    /// Use this when you have already concatenated all fields into a single text string.
    ///
    /// # Arguments
    /// * `doc_ids` - List of document IDs
    /// * `texts` - List of text content, same length as doc_ids
    ///
    /// # Example
    /// ```python
    /// # If you have pre-concatenated text
    /// doc_ids = [1, 2, 3]
    /// texts = ["Title1 Content1", "Title2 Content2", "Title3 Content3"]
    /// engine.add_documents_texts(doc_ids, texts)
    /// 
    /// # Or from DataFrame with combined column
    /// df['combined'] = df['title'] + ' ' + df['content']
    /// engine.add_documents_texts(df['id'].tolist(), df['combined'].tolist())
    /// ```
    #[pyo3(name = "add_documents_texts")]
    fn add_documents_texts_py(&self, doc_ids: Vec<u32>, texts: Vec<String>) -> pyo3::PyResult<usize> {
        self.add_documents_texts(doc_ids, texts).map_err(Into::into)
    }
    
    #[pyo3(name = "update_document")]
    fn update_document_py(&self, doc_id: u32, fields: HashMap<String, String>) -> pyo3::PyResult<()> {
        self.update_document(doc_id, fields).map_err(Into::into)
    }
    
    #[pyo3(name = "remove_document")]
    fn remove_document_py(&self, doc_id: u32) -> pyo3::PyResult<()> {
        self.remove_document(doc_id).map_err(Into::into)
    }
    
    #[pyo3(name = "remove_documents")]
    fn remove_documents_py(&self, doc_ids: Vec<u32>) -> pyo3::PyResult<()> {
        self.remove_documents(doc_ids).map_err(Into::into)
    }
    
    #[pyo3(name = "search")]
    fn search_py(&self, query: &str) -> pyo3::PyResult<ResultHandle> {
        self.search(query).map_err(Into::into)
    }
    
    #[pyo3(signature = (query, min_results=5))]
    #[pyo3(name = "fuzzy_search")]
    fn fuzzy_search_py(&self, query: &str, min_results: usize) -> pyo3::PyResult<ResultHandle> {
        self.fuzzy_search(query, min_results).map_err(Into::into)
    }
    
    #[pyo3(name = "search_batch")]
    fn search_batch_py(&self, queries: Vec<String>) -> pyo3::PyResult<Vec<ResultHandle>> {
        self.search_batch(queries).map_err(Into::into)
    }
    
    #[pyo3(name = "search_and")]
    fn search_and_py(&self, queries: Vec<String>) -> pyo3::PyResult<ResultHandle> {
        self.search_and(queries).map_err(Into::into)
    }
    
    #[pyo3(name = "search_or")]
    fn search_or_py(&self, queries: Vec<String>) -> pyo3::PyResult<ResultHandle> {
        self.search_or(queries).map_err(Into::into)
    }
    
    #[pyo3(name = "filter_by_ids")]
    fn filter_by_ids_py(&self, result: &ResultHandle, allowed_ids: Vec<u32>) -> pyo3::PyResult<ResultHandle> {
        self.filter_by_ids(result, allowed_ids).map_err(Into::into)
    }
    
    #[pyo3(name = "exclude_ids")]
    fn exclude_ids_py(&self, result: &ResultHandle, excluded_ids: Vec<u32>) -> pyo3::PyResult<ResultHandle> {
        self.exclude_ids(result, excluded_ids).map_err(Into::into)
    }
    
    #[pyo3(signature = (threshold=0.7, max_distance=2, max_candidates=20))]
    #[pyo3(name = "set_fuzzy_config")]
    fn set_fuzzy_config_py(&self, threshold: f64, max_distance: usize, max_candidates: usize) -> pyo3::PyResult<()> {
        self.set_fuzzy_config(threshold, max_distance, max_candidates);
        Ok(())
    }
    
    #[pyo3(name = "get_fuzzy_config")]
    fn get_fuzzy_config_py(&self) -> HashMap<String, f64> {
        self.get_fuzzy_config()
    }
    
    #[pyo3(name = "flush")]
    fn flush_py(&self) -> pyo3::PyResult<usize> {
        self.flush().map_err(Into::into)
    }

    #[pyo3(name = "flush_async")]
    fn flush_async_py(&self) -> pyo3::PyResult<()> {
        self.flush_async().map_err(Into::into)
    }

    #[pyo3(name = "wait_flush")]
    fn wait_flush_py(&self) -> pyo3::PyResult<usize> {
        self.wait_flush().map_err(Into::into)
    }
    
    #[pyo3(name = "save")]
    fn save_py(&self) -> pyo3::PyResult<usize> {
        self.save().map_err(Into::into)
    }
    
    #[pyo3(name = "load")]
    fn load_py(&self) -> pyo3::PyResult<()> {
        self.load().map_err(Into::into)
    }
    
    #[pyo3(name = "preload")]
    fn preload_py(&self) -> pyo3::PyResult<usize> {
        self.preload().map_err(Into::into)
    }
    
    #[pyo3(name = "compact")]
    fn compact_py(&self) -> pyo3::PyResult<()> {
        self.compact().map_err(Into::into)
    }
    
    #[pyo3(name = "clear_cache")]
    fn clear_cache_py(&self) {
        self.clear_cache()
    }
    
    #[pyo3(name = "clear_doc_terms")]
    fn clear_doc_terms_py(&self) {
        self.clear_doc_terms()
    }
    
    #[pyo3(name = "is_memory_only")]
    fn is_memory_only_py(&self) -> bool {
        self.is_memory_only()
    }
    
    #[pyo3(name = "term_count")]
    fn term_count_py(&self) -> usize {
        self.term_count()
    }
    
    #[pyo3(name = "buffer_size")]
    fn buffer_size_py(&self) -> usize {
        self.buffer_size()
    }
    
    #[pyo3(name = "doc_terms_size")]
    fn doc_terms_size_py(&self) -> usize {
        self.doc_terms_size()
    }
    
    #[pyo3(name = "page_count")]
    fn page_count_py(&self) -> usize {
        self.page_count()
    }
    
    #[pyo3(name = "segment_count")]
    fn segment_count_py(&self) -> usize {
        self.segment_count()
    }
    
    #[pyo3(name = "memtable_size")]
    fn memtable_size_py(&self) -> usize {
        self.memtable_size()
    }
    
    #[pyo3(name = "stats")]
    fn stats_py(&self) -> HashMap<String, f64> {
        self.stats()
    }
    
    #[pyo3(name = "deleted_count")]
    fn deleted_count_py(&self) -> usize {
        self.deleted_count()
    }
    
    #[pyo3(name = "is_lazy_load")]
    fn is_lazy_load_py(&self) -> bool {
        self.is_lazy_load()
    }
    
    #[pyo3(name = "warmup_terms")]
    fn warmup_terms_py(&self, terms: Vec<String>) -> usize {
        self.warmup_terms(terms)
    }
    
    #[pyo3(name = "clear_lru_cache")]
    fn clear_lru_cache_py(&self) {
        self.clear_lru_cache()
    }
    
    fn __repr__(&self) -> String {
        self.to_string()
    }
}

// ==================== Internal Methods ====================

/// Thread-safe helper: get existing term ID or insert new one, with thread-local cache.
#[inline]
fn get_or_insert_term_id(
    term: &str,
    dict: &dashmap::DashMap<String, u32>,
    next_id: &std::sync::atomic::AtomicU32,
    local_cache: &mut FxHashMap<String, u32>,
) -> u32 {
    if let Some(&id) = local_cache.get(term) {
        return id;
    }
    let id = match dict.entry(term.to_string()) {
        dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
        dashmap::mapref::entry::Entry::Vacant(e) => {
            let new_id = next_id.fetch_add(1, Ordering::Relaxed);
            e.insert(new_id);
            new_id
        }
    };
    local_cache.insert(term.to_string(), id);
    id
}

impl UnifiedEngine {
    /// High-performance batch parallel document processing using ForkUnion
    /// 
    /// Optimization strategy:
    /// 1. Parallel tokenization using ForkUnion thread pool
    /// 2. Thread-local HashMap collection to avoid lock contention
    /// 3. Single-pass merge into DashMap buffer
    /// 4. Optimized string handling to reduce allocations
    fn add_documents_batch_parallel(&self, docs: &[(u32, HashMap<String, String>)]) {
        let num_docs = docs.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let max_chinese_len = self.max_chinese_length;
        let min_term_len = self.min_term_length;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let docs_slice = &docs[start..end];
                s.spawn(move || {
                    let mut text_buffer = String::with_capacity(256);
                    let mut word_buf: Vec<u8> = Vec::with_capacity(256);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(4096, Default::default());
                    let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
                    for (doc_id, fields) in docs_slice {
                        text_buffer.clear();
                        for (idx, value) in fields.values().enumerate() {
                            if idx > 0 { text_buffer.push(' '); }
                            text_buffer.push_str(value);
                        }
                        Self::tokenize_fast_cb(&text_buffer, max_chinese_len, min_term_len, &mut word_buf, |term| {
                            let lid = if let Some(&id) = local_dict.get(term) { id } else {
                                let id = local_terms.len() as u32;
                                local_dict.insert(term.to_string(), id);
                                local_terms.push(Vec::new()); id
                            };
                            local_terms[lid as usize].push(*doc_id);
                        });
                    }
                    let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
                    for (term, &lid) in local_dict.iter() {
                        let gid = match d2.entry(term.to_string()) {
                            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                            dashmap::mapref::entry::Entry::Vacant(e) => {
                                let id = n2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                e.insert(id);
                                i2.insert(id, term.to_string()); id
                            }
                        };
                        l2g[lid as usize] = gid;
                    }
                    for (lid, doc_ids) in local_terms.iter().enumerate() {
                        if !doc_ids.is_empty() {
                            b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                        }
                    }
                });
            }
        });
    }

    fn add_documents_columnar_parallel(&self, doc_ids: &[u32], columns: &[(String, Vec<String>)]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let max_chinese_len = self.max_chinese_length;
        let min_term_len = self.min_term_length;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                s.spawn(move || {
                    let mut text_buffer = String::with_capacity(512);
                    let mut word_buf: Vec<u8> = Vec::with_capacity(256);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(4096, Default::default());
                    let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
                    for idx in start..end {
                        let doc_id = doc_ids[idx];
                        text_buffer.clear();
                        for (col_idx, (_, values)) in columns.iter().enumerate() {
                            if col_idx > 0 { text_buffer.push(' '); }
                            text_buffer.push_str(&values[idx]);
                        }
                        Self::tokenize_fast_cb(&text_buffer, max_chinese_len, min_term_len, &mut word_buf, |term| {
                            let lid = if let Some(&id) = local_dict.get(term) { id } else {
                                let id = local_terms.len() as u32;
                                local_dict.insert(term.to_string(), id);
                                local_terms.push(Vec::new()); id
                            };
                            local_terms[lid as usize].push(doc_id);
                        });
                    }
                    let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
                    for (term, &lid) in local_dict.iter() {
                        let gid = match d2.entry(term.to_string()) {
                            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                            dashmap::mapref::entry::Entry::Vacant(e) => {
                                let id = n2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                e.insert(id);
                                i2.insert(id, term.to_string()); id
                            }
                        };
                        l2g[lid as usize] = gid;
                    }
                    for (lid, doc_ids) in local_terms.iter().enumerate() {
                        if !doc_ids.is_empty() {
                            b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                        }
                    }
                });
            }
        });
    }

    fn add_documents_texts_parallel(&self, doc_ids: &[u32], texts: &[String]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let max_chinese_len = self.max_chinese_length;
        let min_term_len = self.min_term_length;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                s.spawn(move || {
                    let mut word_buf: Vec<u8> = Vec::with_capacity(256);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(4096, Default::default());
                    let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
                    for idx in start..end {
                        let doc_id = doc_ids[idx];
                        Self::tokenize_fast_cb(&texts[idx], max_chinese_len, min_term_len, &mut word_buf, |term| {
                            let lid = if let Some(&id) = local_dict.get(term) { id } else {
                                let id = local_terms.len() as u32;
                                local_dict.insert(term.to_string(), id);
                                local_terms.push(Vec::new()); id
                            };
                            local_terms[lid as usize].push(doc_id);
                        });
                    }
                    let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
                    for (term, &lid) in local_dict.iter() {
                        let gid = match d2.entry(term.to_string()) {
                            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                            dashmap::mapref::entry::Entry::Vacant(e) => {
                                let id = n2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                e.insert(id);
                                i2.insert(id, term.to_string()); id
                            }
                        };
                        l2g[lid as usize] = gid;
                    }
                    for (lid, doc_ids) in local_terms.iter().enumerate() {
                        if !doc_ids.is_empty() {
                            b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                        }
                    }
                });
            }
        });
    }

    fn add_documents_arrow_parallel<'a>(&self, doc_ids: &[u32], columns: &[(String, Vec<&'a str>)]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let max_chinese_len = self.max_chinese_length;
        let min_term_len = self.min_term_length;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let columns_slices: Vec<Vec<&'a str>> = columns.iter().map(|(_, v)| v.to_vec()).collect();
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let thread_columns: Vec<Vec<&'a str>> = columns_slices.iter()
                    .map(|v| v[start..end].to_vec()).collect();
                s.spawn(move || {
                    let mut text_buffer = String::with_capacity(512);
                    let mut word_buf: Vec<u8> = Vec::with_capacity(256);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(4096, Default::default());
                    let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
                    for local_idx in 0..(end - start) {
                        let doc_id = doc_ids[start + local_idx];
                        text_buffer.clear();
                        for (col_idx, col_values) in thread_columns.iter().enumerate() {
                            if col_idx > 0 { text_buffer.push(' '); }
                            text_buffer.push_str(col_values[local_idx]);
                        }
                        Self::tokenize_fast_cb(&text_buffer, max_chinese_len, min_term_len, &mut word_buf, |term| {
                            let lid = if let Some(&id) = local_dict.get(term) { id } else {
                                let id = local_terms.len() as u32;
                                local_dict.insert(term.to_string(), id);
                                local_terms.push(Vec::new()); id
                            };
                            local_terms[lid as usize].push(doc_id);
                        });
                    }
                    let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
                    for (term, &lid) in local_dict.iter() {
                        let gid = match d2.entry(term.to_string()) {
                            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                            dashmap::mapref::entry::Entry::Vacant(e) => {
                                let id = n2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                e.insert(id);
                                i2.insert(id, term.to_string()); id
                            }
                        };
                        l2g[lid as usize] = gid;
                    }
                    for (lid, doc_ids) in local_terms.iter().enumerate() {
                        if !doc_ids.is_empty() {
                            b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                        }
                    }
                });
            }
        });
    }

    /// Zero-copy parallel processing for Arrow single text column
    fn add_documents_arrow_texts_parallel<'a>(&self, doc_ids: &[u32], texts: &[&'a str]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let max_chinese_len = self.max_chinese_length;
        let min_term_len = self.min_term_length;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let texts_slice = &texts[start..end];
                let doc_ids_slice = &doc_ids[start..end];
                s.spawn(move || {
                    let mut word_buf: Vec<u8> = Vec::with_capacity(256);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(4096, Default::default());
                    let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
                    for (local_idx, text) in texts_slice.iter().enumerate() {
                        let doc_id = doc_ids_slice[local_idx];
                        Self::tokenize_fast_cb(text, max_chinese_len, min_term_len, &mut word_buf, |term| {
                            let lid = if let Some(&id) = local_dict.get(term) { id } else {
                                let id = local_terms.len() as u32;
                                local_dict.insert(term.to_string(), id);
                                local_terms.push(Vec::new()); id
                            };
                            local_terms[lid as usize].push(doc_id);
                        });
                    }
                    let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
                    for (term, &lid) in local_dict.iter() {
                        let gid = match d2.entry(term.to_string()) {
                            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                            dashmap::mapref::entry::Entry::Vacant(e) => {
                                let id = n2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                e.insert(id);
                                i2.insert(id, term.to_string()); id
                            }
                        };
                        l2g[lid as usize] = gid;
                    }
                    for (lid, doc_ids) in local_terms.iter().enumerate() {
                        if !doc_ids.is_empty() {
                            b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                        }
                    }
                });
            }
        });
    }

    /// Ultra-fast tokenization function optimized for batch processing
    /// Uses inline Chinese detection to avoid regex overhead
    /// Uses FxHashSet for O(1) deduplication instead of O(n log n) sort+dedup
    #[inline]
    fn tokenize_fast(text: &str, _pattern: &regex::Regex, max_chinese_length: usize, min_term_length: usize) -> Vec<String> {
        // Use FxHashSet for faster deduplication
        let mut terms_set: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        
        // Process character by character without collecting into Vec first
        let mut chars_iter = text.chars().peekable();
        let mut char_buf: Vec<char> = Vec::with_capacity(128);
        
        while let Some(c) = chars_iter.next() {
            let c_lower = c.to_ascii_lowercase();
            
            // Fast inline Chinese character detection
            if Self::is_chinese_char(c) {
                // Collect Chinese sequence
                    char_buf.clear();
                char_buf.push(c);
                while chars_iter.peek().map_or(false, |&next| Self::is_chinese_char(next)) {
                    char_buf.push(chars_iter.next().unwrap());
                }
                
                let chinese_len = char_buf.len();
                
                // Generate n-grams for Chinese
                if chinese_len >= 2 {
                    for n in 2..=max_chinese_length.min(chinese_len) {
                        for j in 0..=chinese_len.saturating_sub(n) {
                            let term: String = char_buf[j..j + n].iter().collect();
                            terms_set.insert(term);
                        }
                    }
                }
            } else if c_lower.is_alphanumeric() {
                // English/numeric word
                char_buf.clear();
                char_buf.push(c_lower);
                while chars_iter.peek().map_or(false, |&next| {
                    let next_lower = next.to_ascii_lowercase();
                    next_lower.is_alphanumeric() && !Self::is_chinese_char(next)
                }) {
                    char_buf.push(chars_iter.next().unwrap().to_ascii_lowercase());
                }
                if char_buf.len() >= min_term_length {
                    let term: String = char_buf.iter().collect();
                    terms_set.insert(term);
                }
            }
            // Skip non-alphanumeric characters implicitly
        }
        
        // Convert to Vec
        terms_set.into_iter().collect()
    }
    
    /// Fast inline Chinese character detection
    #[inline(always)]
    fn is_chinese_char(c: char) -> bool {
        matches!(c, '\u{4e00}'..='\u{9fff}')
    }

    /// Zero-allocation callback tokenizer.
    /// Emits &str slices (from original text for Chinese n-grams, from word_buf for ASCII).
    /// word_buf is a reusable scratch buffer for lowercase ASCII tokens.
    #[inline]
    fn tokenize_fast_cb(
        text: &str,
        max_chinese_length: usize,
        min_term_length: usize,
        word_buf: &mut Vec<u8>,
        mut on_term: impl FnMut(&str),
    ) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            let b = bytes[i];
            if b < 0x80 {
                if b.is_ascii_alphanumeric() {
                    word_buf.clear();
                    let word_start = i;
                    let mut needs_lower = false;
                    while i < len && bytes[i] < 0x80 && bytes[i].is_ascii_alphanumeric() {
                        let c = bytes[i];
                        needs_lower |= c.is_ascii_uppercase();
                        word_buf.push(c.to_ascii_lowercase());
                        i += 1;
                    }
                    if word_buf.len() >= min_term_length {
                        if needs_lower {
                            // SAFETY: word_buf contains only ASCII lowercase bytes
                            on_term(unsafe { std::str::from_utf8_unchecked(word_buf) });
                        } else {
                            on_term(&text[word_start..i]);
                        }
                    }
                } else {
                    i += 1;
                }
            } else if b >= 0xE4 && b <= 0xE9 && i + 2 < len {
                // Possible CJK 3-byte UTF-8
                let mut char_byte_offsets = [0usize; 64];
                let mut n_chars = 0usize;
                let seq_start = i;
                while i < len && n_chars < 64 {
                    let b0 = bytes[i];
                    if b0 >= 0xE4 && b0 <= 0xE9 && i + 2 < len {
                        let b1 = bytes[i + 1];
                        let b2 = bytes[i + 2];
                        let cp = ((b0 as u32 & 0x0F) << 12)
                            | ((b1 as u32 & 0x3F) << 6)
                            | (b2 as u32 & 0x3F);
                        if cp >= 0x4E00 && cp <= 0x9FFF {
                            char_byte_offsets[n_chars] = i - seq_start;
                            n_chars += 1;
                            i += 3;
                            continue;
                        }
                    }
                    break;
                }
                if n_chars >= 2 {
                    for n in 2..=max_chinese_length.min(n_chars) {
                        for j in 0..=n_chars.saturating_sub(n) {
                            let s = seq_start + char_byte_offsets[j];
                            let e = if j + n < n_chars {
                                seq_start + char_byte_offsets[j + n]
                            } else {
                                i
                            };
                            on_term(&text[s..e]);
                        }
                    }
                } else if n_chars == 0 {
                    i += 3;
                }
            } else if b < 0xE0 {
                i += 2;
            } else if b < 0xF0 {
                i += 3;
            } else {
                i += 4;
            }
        }
    }
    
    
    fn add_text(&self, doc_id: u32, text: &str) {
        let terms = Self::tokenize_fast(text, &self.chinese_pattern, self.max_chinese_length, self.min_term_length);
        
        let mut term_ids = if self.track_doc_terms { Vec::with_capacity(terms.len()) } else { Vec::new() };
        
        for term in &terms {
            let term_id = self.get_term_id(term);
            if self.track_doc_terms {
                term_ids.push(term_id);
            }
            self.buffer.entry(term_id)
                .or_insert_with(FastBitmap::new)
                .add(doc_id);
        }
        
        if self.track_doc_terms {
            self.doc_terms.insert(doc_id, term_ids);
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
                    let doc_term_set: std::collections::HashSet<u32> = terms.iter().copied().collect();
                    let mut all_match = true;
                    for word in &words {
                        let mut word_matched = false;
                        if let Some(id_ref) = self.dict.get(&**word) {
                            if doc_term_set.contains(&*id_ref) {
                                word_matched = true;
                            }
                        }
                        
                        if !word_matched {
                            for &term_id in &doc_term_set {
                                if let Some(term_str) = self.get_term_str(term_id) {
                                    if term_str.contains(word) {
                                        word_matched = true;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if !word_matched {
                            all_match = false;
                            break;
                        }
                    }
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
        if let Some(bitmap) = self.buffer.get(&self.get_term_id(term)) {
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
                        
                        if let Some(ngram_id) = self.dict.get(&ngram) {
                            if let Some(bitmap) = self.buffer.get(&*ngram_id) {
                                if result.is_empty() {
                                    result = bitmap.clone();
                                } else {
                                    result.and_inplace(&bitmap);
                                }
                            }
                        }
                        
                        if let Some(ref index) = self.index {
                            if let Some(bitmap) = index.get(ngram.as_str()) {
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
        let mut all_terms: Vec<String> = self.dict.iter()
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

/// Create unified search engine (Rust API)
///
/// # Example
///
/// ```rust,no_run
/// use nanofts::{create_engine, EngineConfig};
///
/// // Memory-only mode
/// let engine = create_engine(EngineConfig::memory_only()).unwrap();
///
/// // Persistent mode with lazy load
/// let engine = create_engine(
///     EngineConfig::persistent("index.nfts")
///         .with_lazy_load(true)
/// ).unwrap();
/// ```
pub fn create_engine(config: EngineConfig) -> EngineResult<UnifiedEngine> {
    UnifiedEngine::new(config)
}

/// Python-specific create_engine function
#[cfg(feature = "python")]
#[pyo3::pyfunction]
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
pub fn create_engine_py(
    index_file: String,
    max_chinese_length: usize,
    min_term_length: usize,
    fuzzy_threshold: f64,
    fuzzy_max_distance: usize,
    track_doc_terms: bool,
    drop_if_exists: bool,
    lazy_load: bool,
    cache_size: usize,
) -> pyo3::PyResult<UnifiedEngine> {
    UnifiedEngine::new_with_params(
        index_file,
        max_chinese_length,
        min_term_length,
        fuzzy_threshold,
        fuzzy_max_distance,
        track_doc_terms,
        drop_if_exists,
        lazy_load,
        cache_size,
    ).map_err(Into::into)
}
