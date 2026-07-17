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

use crate::analyzer::{self, AnalyzedDocument, AnalyzerConfig};
use crate::bitmap::FastBitmap;
use crate::doc_tokens::{DocTokenStore, PackedDocTokens, RankedHit};
use crate::lsm_single::LsmSingleIndex;
use crate::query;
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
    pub fn contains(&self, doc_id: u64) -> bool {
        self.bitmap.contains(doc_id)
    }
    
    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }
    
    /// Get top N document IDs
    pub fn top(&self, n: usize) -> Vec<u64> {
        self.bitmap.iter().take(n).collect()
    }
    
    /// Convert to vector of document IDs
    pub fn to_list(&self) -> Vec<u64> {
        self.bitmap.to_vec()
    }
    
    /// Get page of results
    pub fn page(&self, offset: usize, limit: usize) -> Vec<u64> {
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
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
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
    fn contains_py(&self, doc_id: u64) -> bool {
        self.contains(doc_id)
    }
    
    #[pyo3(name = "is_empty")]
    fn is_empty_py(&self) -> bool {
        self.is_empty()
    }
    
    #[pyo3(signature = (n=100))]
    #[pyo3(name = "top")]
    fn top_py(&self, n: usize) -> Vec<u64> {
        self.top(n)
    }
    
    #[pyo3(name = "to_list")]
    fn to_list_py(&self) -> Vec<u64> {
        self.to_list()
    }
    
    fn to_numpy<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<u64>>> {
        let ids: Vec<u64> = self.bitmap.to_vec();
        Ok(numpy::PyArray1::from_vec_bound(py, ids))
    }
    
    #[pyo3(name = "page")]
    fn page_py(&self, offset: usize, limit: usize) -> Vec<u64> {
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
    /// Whether to track document terms (for efficient deletion, phrase queries, and BM25 ranking)
    pub track_doc_terms: bool,
    /// Whether to delete existing index file
    pub drop_if_exists: bool,
    /// Whether to enable lazy load mode
    pub lazy_load: bool,
    /// LRU cache size in lazy load mode
    pub cache_size: usize,
    /// Maximum number of fuzzy-match term candidates to consider per query word
    pub fuzzy_max_candidates: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            index_file: String::new(),
            max_chinese_length: 3,
            min_term_length: 2,
            fuzzy_threshold: 0.7,
            fuzzy_max_distance: 2,
            track_doc_terms: true,
            drop_if_exists: false,
            lazy_load: false,
            cache_size: 10000,
            fuzzy_max_candidates: 20,
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
    // Index file path (empty in memory-only mode); used to derive the `.tok` sidecar path.
    index_file: String,
    // Memory buffer (memory-only mode or write buffer)
    buffer: DashMap<u32, FastBitmap>,
    // Document-term mapping (for efficient deletion): doc_id -> term_ids
    doc_terms: DashMap<u64, Vec<u32>>,
    track_doc_terms: bool,
    // Per-document token streams (global term ids) for phrase queries and BM25 ranking.
    token_store: RwLock<DocTokenStore>,
    // Deletion markers (tombstone): docs removed from live results entirely.
    deleted_docs: RwLock<FastBitmap>,
    // Shadow markers (ApexFTS-style): base (index) hits for these docs must be ignored,
    // because the buffer (delta) holds their up-to-date content. Updated docs are shadowed
    // (not removed), deleted docs are both shadowed and deleted.
    shadowed_docs: RwLock<FastBitmap>,
    // Configuration
    max_chinese_length: usize,
    // Analyzer configuration derived from max_chinese_length/min_term_length.
    analyzer_config: AnalyzerConfig,
    fuzzy_config: RwLock<FuzzyConfig>,
    // Cache
    result_cache: DashMap<String, Arc<FastBitmap>>,
    cache_enabled: std::sync::atomic::AtomicBool,
    // Statistics
    stats: EngineStats,
    // Mode
    memory_only: bool,
    lazy_load: bool,
    preloaded: std::sync::atomic::AtomicBool,
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
    
    // ==================== Shadow/Delete Marker Helpers ====================
    // ApexFTS-style base/delta shadow semantics:
    // - `shadowed_docs`: base (on-disk index) postings for these doc ids must be ignored
    //   because the buffer (delta) holds their latest content (or they were deleted).
    // - `deleted_docs`: doc ids removed from live results entirely.
    
    /// Mark a document as shadowed (its base/index postings must be ignored).
    #[inline]
    fn mark_shadowed(&self, doc_id: u64) {
        self.shadowed_docs.write().add(doc_id);
    }
    
    /// Mark a document as deleted (removed from live results).
    #[inline]
    fn mark_deleted(&self, doc_id: u64) {
        self.deleted_docs.write().add(doc_id);
    }
    
    /// Clear the shadow marker for a document (its base postings are trustworthy again).
    #[inline]
    fn clear_shadowed(&self, doc_id: u64) {
        self.shadowed_docs.write().remove(doc_id);
    }
    
    /// Clear the delete marker for a document (it is live again).
    #[inline]
    fn clear_deleted(&self, doc_id: u64) {
        self.deleted_docs.write().remove(doc_id);
    }
    
    /// Clear both shadow and delete markers for a batch of doc ids (used when (re-)adding
    /// documents, since a freshly-(re)added document is live and its indexed base data,
    /// if any, is about to be superseded by the new buffer content anyway).
    fn clear_markers_for_docs(&self, doc_ids: &[u64]) {
        if doc_ids.is_empty() {
            return;
        }
        {
            let mut deleted = self.deleted_docs.write();
            if !deleted.is_empty() {
                for &doc_id in doc_ids {
                    deleted.remove(doc_id);
                }
            }
        }
        {
            let mut shadowed = self.shadowed_docs.write();
            if !shadowed.is_empty() {
                for &doc_id in doc_ids {
                    shadowed.remove(doc_id);
                }
            }
        }
    }
    
    /// Drain the in-memory buffer into a Vec of (term_string, bitmap) pairs,
    /// returning the set of doc IDs that were flushed.
    fn drain_buffer(&self) -> (Vec<(String, FastBitmap)>, std::collections::HashSet<u64>) {
        let mut entries: Vec<(String, FastBitmap)> = Vec::new();
        let mut flushed_doc_ids = std::collections::HashSet::new();
        let keys: Vec<u32> = self.buffer.iter().map(|e| *e.key()).collect();
        for key in keys {
            if let Some((term_id, bitmap)) = self.buffer.remove(&key) {
                if let Some(term_str) = self.id_to_str.get(&term_id) {
                    for doc_id in bitmap.iter() {
                        flushed_doc_ids.insert(doc_id);
                    }
                    entries.push((term_str.clone(), bitmap));
                }
            }
        }
        (entries, flushed_doc_ids)
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
                let _ = std::fs::remove_file(format!("{}.tok", config.index_file));
                LsmSingleIndex::create_full_options(&path, true, config.lazy_load, config.cache_size)
            } else if path.exists() {
                LsmSingleIndex::open_full_options(&path, true, config.lazy_load, config.cache_size)
            } else {
                LsmSingleIndex::create_full_options(&path, true, config.lazy_load, config.cache_size)
            }.map_err(|e| EngineError::IndexError(e.to_string()))?;
            
            Some(Arc::new(idx))
        };
        
        let engine = Self {
            dict: Arc::new(dashmap::DashMap::with_shard_amount(64)),
            id_to_str: Arc::new(dashmap::DashMap::with_shard_amount(64)),
            next_term_id: std::sync::atomic::AtomicU32::new(1),
            index,
            index_file: config.index_file.clone(),
            buffer: DashMap::with_shard_amount(64),
            doc_terms: DashMap::new(),
            track_doc_terms: config.track_doc_terms,
            token_store: RwLock::new(DocTokenStore::new()),
            deleted_docs: RwLock::new(FastBitmap::new()),
            shadowed_docs: RwLock::new(FastBitmap::new()),
            max_chinese_length: config.max_chinese_length,
            analyzer_config: AnalyzerConfig {
                max_chinese_length: config.max_chinese_length,
                min_term_length: config.min_term_length,
            },
            fuzzy_config: RwLock::new(FuzzyConfig {
                threshold: config.fuzzy_threshold,
                max_distance: config.fuzzy_max_distance,
                max_candidates: config.fuzzy_max_candidates,
            }),
            result_cache: DashMap::new(),
            cache_enabled: std::sync::atomic::AtomicBool::new(true),
            stats: EngineStats::default(),
            memory_only,
            lazy_load: config.lazy_load,
            preloaded: std::sync::atomic::AtomicBool::new(false),
            compact_lock: RwLock::new(()),
            flush_handle: Mutex::new(None),
        };
        
        engine.load_token_store()?;
        
        Ok(engine)
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
            fuzzy_max_candidates: 20,
        })
    }
    
    // ==================== Document Operations ====================
    
    /// Add single document
    pub fn add_document(&self, doc_id: u64, fields: HashMap<String, String>) -> EngineResult<()> {
        // Document is live again: clear any prior delete/shadow markers.
        self.clear_deleted(doc_id);
        self.clear_shadowed(doc_id);
        
        let text: String = fields.values().cloned().collect::<Vec<_>>().join(" ");
        self.add_text(doc_id, &text);
        
        // Clear result cache
        self.result_cache.clear();
        Ok(())
    }
    
    /// Batch add documents - optimized parallel version using ForkUnion
    pub fn add_documents(&self, docs: Vec<(u64, HashMap<String, String>)>) -> EngineResult<usize> {
        let count = docs.len();
        if count == 0 {
            return Ok(0);
        }
        
        // Clear deleted/shadowed markers for all docs (live again)
        let doc_ids: Vec<u64> = docs.iter().map(|(id, _)| *id).collect();
        self.clear_markers_for_docs(&doc_ids);
        
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
        doc_ids: Vec<u64>, 
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
        
        // Clear deleted/shadowed markers for all docs (live again)
        self.clear_markers_for_docs(&doc_ids);
        
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
    pub fn add_documents_texts(&self, doc_ids: Vec<u64>, texts: Vec<String>) -> EngineResult<usize> {
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
        
        // Clear deleted/shadowed markers for all docs (live again)
        self.clear_markers_for_docs(&doc_ids);
        
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
    /// * `doc_ids` - Document IDs as u64 slice
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
        doc_ids: &[u64],
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
        
        // Clear deleted/shadowed markers for all docs (live again)
        self.clear_markers_for_docs(doc_ids);
        
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
    /// * `doc_ids` - Document IDs as u64 slice
    /// * `texts` - Text content as string slices from Arrow StringArray
    pub fn add_documents_arrow_texts<'a>(
        &self,
        doc_ids: &[u64],
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
        
        // Clear deleted/shadowed markers for all docs (live again)
        self.clear_markers_for_docs(doc_ids);
        
        // Use optimized zero-copy single-column processing
        self.add_documents_arrow_texts_parallel(doc_ids, texts);
        
        self.result_cache.clear();
        Ok(num_docs)
    }
    
    /// Update document
    pub fn update_document(&self, doc_id: u64, fields: HashMap<String, String>) -> EngineResult<()> {
        // 1. Shadow the document: base (index) postings for it must be ignored during
        //    search until its new content is flushed into the base. This makes it safe
        //    to let the doc live in the buffer only, without any base/buffer coupling.
        self.mark_shadowed(doc_id);
        
        // 2. Ensure document not in deleted_docs (if previously deleted)
        self.clear_deleted(doc_id);
        
        // 3. If track_doc_terms enabled, remove old terms from buffer
        if self.track_doc_terms {
            if let Some((_, terms)) = self.doc_terms.remove(&doc_id) {
                for term in terms {
                    if let Some(mut entry) = self.buffer.get_mut(&term) {
                        entry.remove(doc_id);
                    }
                }
            }
            self.token_store.write().remove(doc_id);
        }
        
        // 4. Add new content to buffer
        let text: String = fields.values().cloned().collect::<Vec<_>>().join(" ");
        self.add_text(doc_id, &text);
        
        // 5. Clear result cache
        self.result_cache.clear();
        
        Ok(())
    }
    
    /// Delete document
    pub fn remove_document(&self, doc_id: u64) -> EngineResult<()> {
        // Shadow the document (ignore any base/index postings for it) AND tombstone it
        // (remove it from live results entirely).
        self.mark_shadowed(doc_id);
        self.mark_deleted(doc_id);
        
        // If track_doc_terms enabled, also remove from buffer (improve memory efficiency)
        if self.track_doc_terms {
            if let Some((_, terms)) = self.doc_terms.remove(&doc_id) {
                for term in terms {
                    if let Some(mut entry) = self.buffer.get_mut(&term) {
                        entry.remove(doc_id);
                    }
                }
            }
            self.token_store.write().remove(doc_id);
        }
        
        self.result_cache.clear();
        Ok(())
    }
    
    /// Batch delete documents
    pub fn remove_documents(&self, doc_ids: Vec<u64>) -> EngineResult<()> {
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
        if self.cache_enabled.load(Ordering::Relaxed) {
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
        
        if self.cache_enabled.load(Ordering::Relaxed) {
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
    
    /// BM25-ranked search, returning up to `limit` hits ordered by descending score.
    ///
    /// Requires `track_doc_terms=true` in [`EngineConfig`] (per-document token streams
    /// must be tracked to compute term frequencies and document lengths).
    pub fn search_ranked(&self, query: &str, limit: usize) -> EngineResult<Vec<RankedHit>> {
        if !self.track_doc_terms {
            return Err(EngineError::InvalidArgument(
                "search_ranked requires track_doc_terms=true".to_string(),
            ));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        
        let (analyzed, postings, candidates) = self.search_candidates(query);
        if analyzed.terms.is_empty() || candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        let store = self.token_store.read();
        let live_doc_count = store.live_doc_count().max(1) as f32;
        let average_length = store.live_term_total() as f32 / live_doc_count;
        
        let mut query_term_ids: Vec<(u32, f32)> = Vec::with_capacity(analyzed.terms.len());
        for (term, posting) in analyzed.terms.iter().zip(postings.iter()) {
            if let Some(gid_ref) = self.dict.get(term.as_str()) {
                let gid = *gid_ref;
                let df = posting.len() as f32;
                query_term_ids.push((gid, query::bm25_idf(live_doc_count, df)));
            }
        }
        if query_term_ids.is_empty() {
            return Ok(Vec::new());
        }
        
        Ok(query::bm25_rank(&candidates, &query_term_ids, &store, average_length, limit))
    }
    
    /// BM25-ranked search returning all matching hits (no limit).
    pub fn search_scored(&self, query: &str) -> EngineResult<Vec<RankedHit>> {
        self.search_ranked(query, usize::MAX)
    }
    
    /// BM25-ranked search returning only the top-N document IDs.
    pub fn search_top_n(&self, query: &str, n: usize) -> EngineResult<Vec<u64>> {
        Ok(self.search_ranked(query, n)?.into_iter().map(|hit| hit.doc_id).collect())
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
    pub fn filter_by_ids(&self, result: &ResultHandle, allowed_ids: Vec<u64>) -> EngineResult<ResultHandle> {
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
    pub fn exclude_ids(&self, result: &ResultHandle, excluded_ids: Vec<u64>) -> EngineResult<ResultHandle> {
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
                // Clean shadow markers for all flushed documents: their latest content is
                // now in the base index, so base postings are trustworthy again.
                {
                    let mut shadowed = self.shadowed_docs.write();
                    if !shadowed.is_empty() {
                        for (_, bitmap) in &entries {
                            for doc_id in bitmap.iter() {
                                shadowed.remove(doc_id);
                            }
                        }
                    }
                }
                index.upsert_batch(entries);
                index.flush()
                    .map_err(|e| EngineError::IndexError(e.to_string()))?;
            }
            
            self.save_token_store()?;
            
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

            // Pre-clean shadow markers for all flushed documents: their latest content is
            // about to become part of the base index (via merge_into_data below).
            {
                let mut shadowed = self.shadowed_docs.write();
                if !shadowed.is_empty() {
                    for (_, bitmap) in &entries {
                        for doc_id in bitmap.iter() {
                            shadowed.remove(doc_id);
                        }
                    }
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
                if result.is_ok() {
                    self.save_token_store()?;
                }
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
            
            let (pending_entries, _flushed_doc_ids) = self.drain_buffer();
            
            // Docs to strip from the rewritten base: tombstoned docs (removed for good) and
            // shadowed docs (their base postings are stale; fresh content lives in the
            // buffer and gets re-upserted right below via pending/extra entries).
            let deleted: Vec<u64> = {
                let mut combined = self.deleted_docs.read().clone();
                combined.or_inplace(&self.shadowed_docs.read());
                combined.to_vec()
            };
            
            index.compact_with_deletions(&deleted)
                .map_err(|e| EngineError::IndexError(e.to_string()))?;
            
            if !pending_entries.is_empty() {
                index.upsert_batch(pending_entries);
                index.flush()
                    .map_err(|e| EngineError::IndexError(e.to_string()))?;
            }
            
            // Drain any entries that arrived during compaction
            let (extra_entries, _extra_doc_ids) = self.drain_buffer();
            if !extra_entries.is_empty() {
                index.upsert_batch(extra_entries);
                index.flush()
                    .map_err(|e| EngineError::IndexError(e.to_string()))?;
            }
            
            // compact_with_deletions rewrote the base cleanly with all tombstoned/shadowed
            // docs stripped, and any fresh buffer content was re-upserted above, so every
            // marker can be reset in one shot.
            *self.deleted_docs.write() = FastBitmap::new();
            *self.shadowed_docs.write() = FastBitmap::new();
            
            self.save_token_store()?;
        }
        Ok(())
    }
    
    /// Compact the token store to a packed base and write it atomically to `{index_file}.tok`.
    /// No-op in memory-only mode or when `track_doc_terms` is disabled.
    fn save_token_store(&self) -> EngineResult<()> {
        if !self.track_doc_terms || self.index_file.is_empty() {
            return Ok(());
        }
        let tok_path = format!("{}.tok", self.index_file);
        let packed = self.token_store.read().compact_to_packed();
        crate::persist::durable_write(std::path::Path::new(&tok_path), &packed.to_bytes())?;
        Ok(())
    }
    
    /// Load a previously-saved `{index_file}.tok` sidecar (if present) into the token store.
    /// No-op in memory-only mode or when `track_doc_terms` is disabled.
    fn load_token_store(&self) -> EngineResult<()> {
        if !self.track_doc_terms || self.index_file.is_empty() {
            return Ok(());
        }
        let tok_path = format!("{}.tok", self.index_file);
        if !std::path::Path::new(&tok_path).exists() {
            return Ok(());
        }
        let bytes = std::fs::read(&tok_path)?;
        let packed = PackedDocTokens::from_bytes(&bytes)
            .map_err(|e| EngineError::IndexError(e.to_string()))?;
        self.token_store.write().replace_base(packed);
        Ok(())
    }
    
    // ==================== Cache Operations ====================
    
    /// Clear result cache
    pub fn clear_cache(&self) {
        self.result_cache.clear();
    }

    /// Enable or disable the query result cache (enabled by default).
    pub fn set_result_cache_enabled(&self, enabled: bool) {
        self.cache_enabled.store(enabled, Ordering::Relaxed);
        if !enabled {
            self.result_cache.clear();
        }
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
    
    /// Get page count (alias for segment_count)
    pub fn page_count(&self) -> usize {
        self.segment_count()
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
        map.insert("deleted_count".to_string(), self.deleted_docs.read().len() as f64);
        map.insert("shadowed_count".to_string(), self.shadowed_docs.read().len() as f64);
        map.insert("memory_only".to_string(), if self.memory_only { 1.0 } else { 0.0 });
        map.insert("lazy_load".to_string(), if self.lazy_load { 1.0 } else { 0.0 });
        map.insert("track_doc_terms".to_string(), if self.track_doc_terms { 1.0 } else { 0.0 });
        
        if let Some(ref index) = self.index {
            map.insert("wal_enabled".to_string(), if index.is_wal_enabled() { 1.0 } else { 0.0 });
            map.insert("wal_size".to_string(), index.wal_size() as f64);
            map.insert("wal_pending_batches".to_string(), index.wal_pending_batches() as f64);
            map.insert("needs_rebuild".to_string(), if index.needs_rebuild() { 1.0 } else { 0.0 });
            
            if index.is_lazy_load() {
                let (lru_hits, lru_misses, lru_size) = index.cache_stats();
                map.insert("lru_cache_hits".to_string(), lru_hits as f64);
                map.insert("lru_cache_misses".to_string(), lru_misses as f64);
                map.insert("lru_cache_size".to_string(), lru_size as f64);
                map.insert("lru_cache_hit_rate".to_string(), index.cache_hit_rate());
                map.insert("term_dir_loaded".to_string(), if index.term_dir_loaded() { 1.0 } else { 0.0 });
            }
        }
        
        map
    }
    
    /// Get deleted document count
    pub fn deleted_count(&self) -> usize {
        self.deleted_docs.read().len() as usize
    }
    
    /// Whether lazy load is enabled
    pub fn is_lazy_load(&self) -> bool {
        self.lazy_load
    }
    
    /// True when the on-disk index recommends a `compact()` call to rebuild its
    /// lazy-load term directory sidecar (missing, stale, or corrupt). Safe to
    /// ignore — the engine transparently falls back to the slower lookup path —
    /// but calling `compact()` restores the faster mmap-backed lazy-load path.
    pub fn needs_rebuild(&self) -> bool {
        self.index.as_ref().is_some_and(|i| i.needs_rebuild())
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
        max_chinese_length=3,
        min_term_length=2,
        fuzzy_threshold=0.7,
        fuzzy_max_distance=2,
        track_doc_terms=true,
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
    fn add_document_py(&self, doc_id: u64, fields: HashMap<String, String>) -> pyo3::PyResult<()> {
        self.add_document(doc_id, fields).map_err(Into::into)
    }
    
    #[pyo3(name = "add_documents")]
    fn add_documents_py(&self, docs: Vec<(u64, HashMap<String, String>)>) -> pyo3::PyResult<usize> {
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
        doc_ids: Vec<u64>, 
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
    fn add_documents_texts_py(&self, doc_ids: Vec<u64>, texts: Vec<String>) -> pyo3::PyResult<usize> {
        self.add_documents_texts(doc_ids, texts).map_err(Into::into)
    }
    
    #[pyo3(name = "update_document")]
    fn update_document_py(&self, doc_id: u64, fields: HashMap<String, String>) -> pyo3::PyResult<()> {
        self.update_document(doc_id, fields).map_err(Into::into)
    }
    
    #[pyo3(name = "remove_document")]
    fn remove_document_py(&self, doc_id: u64) -> pyo3::PyResult<()> {
        self.remove_document(doc_id).map_err(Into::into)
    }
    
    #[pyo3(name = "remove_documents")]
    fn remove_documents_py(&self, doc_ids: Vec<u64>) -> pyo3::PyResult<()> {
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
    
    /// BM25-ranked search returning a list of `(doc_id, score)` tuples ordered by
    /// descending relevance. Requires `track_doc_terms=True`.
    #[pyo3(signature = (query, limit=100))]
    #[pyo3(name = "search_ranked")]
    fn search_ranked_py(&self, query: &str, limit: usize) -> pyo3::PyResult<Vec<(u64, f32)>> {
        Ok(self.search_ranked(query, limit)?
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect())
    }
    
    /// BM25-ranked search returning all matching `(doc_id, score)` tuples (no limit).
    #[pyo3(name = "search_scored")]
    fn search_scored_py(&self, query: &str) -> pyo3::PyResult<Vec<(u64, f32)>> {
        Ok(self.search_scored(query)?
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect())
    }
    
    /// BM25-ranked search returning only the top-N document IDs.
    #[pyo3(signature = (query, n=10))]
    #[pyo3(name = "search_top_n")]
    fn search_top_n_py(&self, query: &str, n: usize) -> pyo3::PyResult<Vec<u64>> {
        self.search_top_n(query, n).map_err(Into::into)
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
    fn filter_by_ids_py(&self, result: &ResultHandle, allowed_ids: Vec<u64>) -> pyo3::PyResult<ResultHandle> {
        self.filter_by_ids(result, allowed_ids).map_err(Into::into)
    }
    
    #[pyo3(name = "exclude_ids")]
    fn exclude_ids_py(&self, result: &ResultHandle, excluded_ids: Vec<u64>) -> pyo3::PyResult<ResultHandle> {
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

    #[pyo3(name = "needs_rebuild")]
    fn needs_rebuild_py(&self) -> bool {
        self.needs_rebuild()
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

/// Estimate unique-term capacity for a worker chunk (high overlap across docs).
#[inline]
fn local_term_capacity(chunk_len: usize) -> usize {
    chunk_len.saturating_mul(4).clamp(256, 65_536)
}

/// Merge thread-local term dictionary and posting lists into global structures.
/// Takes `local_dict` by value so new terms can be moved into the global map
/// (avoids an extra `String` clone on the insert path).
/// Returns local-id → global-id mapping for token remap.
#[inline]
fn merge_local_to_global(
    local_dict: FxHashMap<String, u32>,
    mut local_terms: Vec<Vec<u64>>,
    dict: &dashmap::DashMap<String, u32>,
    id_to_str: &dashmap::DashMap<u32, String>,
    next_term_id: &std::sync::atomic::AtomicU32,
    buffer: &DashMap<u32, FastBitmap>,
) -> Vec<u32> {
    let mut l2g = vec![0u32; local_terms.len()];
    for (term, lid) in local_dict {
        let gid = if let Some(r) = dict.get(term.as_str()) {
            *r.value()
        } else {
            match dict.entry(term) {
                dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
                dashmap::mapref::entry::Entry::Vacant(e) => {
                    let id = next_term_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let owned = e.key().clone();
                    e.insert(id);
                    id_to_str.insert(id, owned);
                    id
                }
            }
        };
        l2g[lid as usize] = gid;
    }
    for (lid, doc_ids) in local_terms.iter_mut().enumerate() {
        if doc_ids.is_empty() {
            continue;
        }
        // Sorted append is much faster than per-id insert in RoaringTreemap::extend.
        doc_ids.sort_unstable();
        let gid = l2g[lid];
        if let Some(mut bm) = buffer.get_mut(&gid) {
            bm.add_many_sorted(doc_ids);
        } else {
            match buffer.entry(gid) {
                dashmap::mapref::entry::Entry::Occupied(mut e) => {
                    e.get_mut().add_many_sorted(doc_ids);
                }
                dashmap::mapref::entry::Entry::Vacant(e) => {
                    e.insert(FastBitmap::from_sorted_slice(doc_ids));
                }
            }
        }
    }
    l2g
}

/// Analyze one document's text, intern its (already-unique) terms into the thread-local
/// dictionary, and — when `track_doc_terms` — record this doc's local term/token ids for
/// later remap to global ids (see [`remap_and_store_tokens`]).
///
/// Takes `AnalyzedDocument` by value so term strings / token vectors can be moved
/// (no per-term `clone()` on the ingest hot path).
#[inline]
fn analyze_and_intern_local(
    doc_id: u64,
    analyzed: AnalyzedDocument,
    local_dict: &mut FxHashMap<String, u32>,
    local_terms: &mut Vec<Vec<u64>>,
    track_doc_terms: bool,
    doc_tokens_local: &mut Vec<(u64, Vec<u32>, Vec<u32>)>,
) {
    if analyzed.terms.is_empty() {
        if track_doc_terms {
            doc_tokens_local.push((doc_id, Vec::new(), Vec::new()));
        }
        return;
    }
    let AnalyzedDocument { terms, tokens } = analyzed;
    let mut term_lids: Vec<u32> = Vec::with_capacity(terms.len());
    for term in terms {
        let lid = if let Some(&id) = local_dict.get(term.as_str()) {
            id
        } else {
            let id = local_terms.len() as u32;
            local_dict.insert(term, id);
            // Docs in a chunk are processed in order; reserve a small run for this term.
            local_terms.push(Vec::with_capacity(8));
            id
        };
        local_terms[lid as usize].push(doc_id);
        term_lids.push(lid);
    }
    if track_doc_terms {
        doc_tokens_local.push((doc_id, term_lids, tokens));
    }
}

/// Remap thread-local (doc_id, term_lids, tokens) records to global term ids using `l2g`
/// (produced by [`merge_local_to_global`]) and write them into `doc_terms` / `token_store`
/// under a single write-lock acquisition.
#[inline]
fn remap_and_store_tokens(
    doc_tokens_local: Vec<(u64, Vec<u32>, Vec<u32>)>,
    l2g: &[u32],
    doc_terms: &DashMap<u64, Vec<u32>>,
    token_store: &RwLock<DocTokenStore>,
) {
    if doc_tokens_local.is_empty() {
        return;
    }
    let mut store = token_store.write();
    for (doc_id, term_lids, tokens) in doc_tokens_local {
        let global_terms: Vec<u32> = term_lids.iter().map(|&lid| l2g[lid as usize]).collect();
        let global_tokens: Vec<u32> = tokens.iter()
            .map(|&t| if t == 0 { 0 } else { global_terms[t as usize - 1] })
            .collect();
        doc_terms.insert(doc_id, global_terms);
        store.upsert(doc_id, global_tokens.into_boxed_slice());
    }
}

impl UnifiedEngine {
    /// High-performance batch parallel document processing using ForkUnion
    /// 
    /// Optimization strategy:
    /// 1. Parallel analysis using ForkUnion thread pool
    /// 2. Thread-local HashMap collection to avoid lock contention
    /// 3. Single-pass merge into DashMap buffer
    /// 4. Optimized string handling to reduce allocations
    fn add_documents_batch_parallel(&self, docs: &[(u64, HashMap<String, String>)]) {
        let num_docs = docs.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let cfg = self.analyzer_config;
        let track_doc_terms = self.track_doc_terms;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        let dt2 = &self.doc_terms; let tk2 = &self.token_store;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let docs_slice = &docs[start..end];
                s.spawn(move || {
                    let cap = local_term_capacity(docs_slice.len());
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(cap, Default::default());
                    let mut local_terms: Vec<Vec<u64>> = Vec::with_capacity(cap);
                    let mut doc_tokens_local = Vec::with_capacity(if track_doc_terms { docs_slice.len() } else { 0 });
                    for (doc_id, fields) in docs_slice {
                        let analyzed = analyzer::analyze_fields(
                            fields.values().map(|s| s.as_str()),
                            &cfg,
                            track_doc_terms,
                        );
                        analyze_and_intern_local(*doc_id, analyzed, &mut local_dict, &mut local_terms, track_doc_terms, &mut doc_tokens_local);
                    }
                    let l2g = merge_local_to_global(local_dict, local_terms, d2, i2, n2, b2);
                    if track_doc_terms {
                        remap_and_store_tokens(doc_tokens_local, &l2g, dt2, tk2);
                    }
                });
            }
        });
    }

    fn add_documents_columnar_parallel(&self, doc_ids: &[u64], columns: &[(String, Vec<String>)]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let cfg = self.analyzer_config;
        let track_doc_terms = self.track_doc_terms;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        let dt2 = &self.doc_terms; let tk2 = &self.token_store;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                s.spawn(move || {
                    let cap = local_term_capacity(end - start);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(cap, Default::default());
                    let mut local_terms: Vec<Vec<u64>> = Vec::with_capacity(cap);
                    let mut doc_tokens_local = Vec::with_capacity(if track_doc_terms { end - start } else { 0 });
                    for idx in start..end {
                        let doc_id = doc_ids[idx];
                        let analyzed = analyzer::analyze_fields(
                            columns.iter().map(|(_, values)| values[idx].as_str()),
                            &cfg,
                            track_doc_terms,
                        );
                        analyze_and_intern_local(doc_id, analyzed, &mut local_dict, &mut local_terms, track_doc_terms, &mut doc_tokens_local);
                    }
                    let l2g = merge_local_to_global(local_dict, local_terms, d2, i2, n2, b2);
                    if track_doc_terms {
                        remap_and_store_tokens(doc_tokens_local, &l2g, dt2, tk2);
                    }
                });
            }
        });
    }

    fn add_documents_texts_parallel(&self, doc_ids: &[u64], texts: &[String]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let cfg = self.analyzer_config;
        let track_doc_terms = self.track_doc_terms;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        let dt2 = &self.doc_terms; let tk2 = &self.token_store;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                s.spawn(move || {
                    let cap = local_term_capacity(end - start);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(cap, Default::default());
                    let mut local_terms: Vec<Vec<u64>> = Vec::with_capacity(cap);
                    let mut doc_tokens_local = Vec::with_capacity(if track_doc_terms { end - start } else { 0 });
                    for idx in start..end {
                        let doc_id = doc_ids[idx];
                        let analyzed = analyzer::analyze_query(&texts[idx], &cfg, track_doc_terms);
                        analyze_and_intern_local(doc_id, analyzed, &mut local_dict, &mut local_terms, track_doc_terms, &mut doc_tokens_local);
                    }
                    let l2g = merge_local_to_global(local_dict, local_terms, d2, i2, n2, b2);
                    if track_doc_terms {
                        remap_and_store_tokens(doc_tokens_local, &l2g, dt2, tk2);
                    }
                });
            }
        });
    }

    fn add_documents_arrow_parallel<'a>(&self, doc_ids: &[u64], columns: &[(String, Vec<&'a str>)]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let cfg = self.analyzer_config;
        let track_doc_terms = self.track_doc_terms;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let columns_slices: Vec<Vec<&'a str>> = columns.iter().map(|(_, v)| v.to_vec()).collect();
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        let dt2 = &self.doc_terms; let tk2 = &self.token_store;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let thread_columns: Vec<Vec<&'a str>> = columns_slices.iter()
                    .map(|v| v[start..end].to_vec()).collect();
                s.spawn(move || {
                    let cap = local_term_capacity(end - start);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(cap, Default::default());
                    let mut local_terms: Vec<Vec<u64>> = Vec::with_capacity(cap);
                    let mut doc_tokens_local = Vec::with_capacity(if track_doc_terms { end - start } else { 0 });
                    for local_idx in 0..(end - start) {
                        let doc_id = doc_ids[start + local_idx];
                        let analyzed = analyzer::analyze_fields(
                            thread_columns.iter().map(|col_values| col_values[local_idx]),
                            &cfg,
                            track_doc_terms,
                        );
                        analyze_and_intern_local(doc_id, analyzed, &mut local_dict, &mut local_terms, track_doc_terms, &mut doc_tokens_local);
                    }
                    let l2g = merge_local_to_global(local_dict, local_terms, d2, i2, n2, b2);
                    if track_doc_terms {
                        remap_and_store_tokens(doc_tokens_local, &l2g, dt2, tk2);
                    }
                });
            }
        });
    }

    /// Zero-copy parallel processing for Arrow single text column
    fn add_documents_arrow_texts_parallel<'a>(&self, doc_ids: &[u64], texts: &[&'a str]) {
        let num_docs = doc_ids.len();
        if num_docs == 0 { return; }
        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let cfg = self.analyzer_config;
        let track_doc_terms = self.track_doc_terms;
        let chunk_size = (num_docs + num_threads - 1) / num_threads;
        let d2 = &self.dict; let i2 = &self.id_to_str; let n2 = &self.next_term_id; let b2 = &self.buffer;
        let dt2 = &self.doc_terms; let tk2 = &self.token_store;
        std::thread::scope(|s| {
            for thread_idx in 0..num_threads {
                let start = thread_idx * chunk_size;
                if start >= num_docs { break; }
                let end = (start + chunk_size).min(num_docs);
                let texts_slice = &texts[start..end];
                let doc_ids_slice = &doc_ids[start..end];
                s.spawn(move || {
                    let cap = local_term_capacity(end - start);
                    let mut local_dict: FxHashMap<String, u32> =
                        FxHashMap::with_capacity_and_hasher(cap, Default::default());
                    let mut local_terms: Vec<Vec<u64>> = Vec::with_capacity(cap);
                    let mut doc_tokens_local = Vec::with_capacity(if track_doc_terms { end - start } else { 0 });
                    for (local_idx, text) in texts_slice.iter().enumerate() {
                        let doc_id = doc_ids_slice[local_idx];
                        let analyzed = analyzer::analyze_query(text, &cfg, track_doc_terms);
                        analyze_and_intern_local(doc_id, analyzed, &mut local_dict, &mut local_terms, track_doc_terms, &mut doc_tokens_local);
                    }
                    let l2g = merge_local_to_global(local_dict, local_terms, d2, i2, n2, b2);
                    if track_doc_terms {
                        remap_and_store_tokens(doc_tokens_local, &l2g, dt2, tk2);
                    }
                });
            }
        });
    }

    fn add_text(&self, doc_id: u64, text: &str) {
        let analyzed = analyzer::analyze_query(text, &self.analyzer_config, self.track_doc_terms);

        if analyzed.terms.is_empty() {
            if self.track_doc_terms {
                self.doc_terms.insert(doc_id, Vec::new());
                self.token_store.write().upsert(doc_id, Box::new([]));
            }
            return;
        }

        let AnalyzedDocument { terms, tokens } = analyzed;
        let mut global_ids: Vec<u32> = Vec::with_capacity(terms.len());
        for term in &terms {
            let gid = self.get_term_id(term);
            self.buffer.entry(gid)
                .or_insert_with(FastBitmap::new)
                .add(doc_id);
            global_ids.push(gid);
        }

        if self.track_doc_terms {
            let global_tokens: Vec<u32> = tokens.iter()
                .map(|&local_id| if local_id == 0 { 0 } else { global_ids[local_id as usize - 1] })
                .collect();
            self.doc_terms.insert(doc_id, global_ids);
            self.token_store.write().upsert(doc_id, global_tokens.into_boxed_slice());
        }
    }
    
    /// Analyze `query`, resolve per-term postings, intersect them, optionally filter by
    /// phrase position, and exclude tombstoned documents. Shared by [`search_internal`]
    /// and the BM25 ranking APIs (which additionally need per-term document frequencies).
    fn search_candidates(&self, query: &str) -> (AnalyzedDocument, Vec<FastBitmap>, FastBitmap) {
        let (inner, is_phrase) = analyzer::phrase_query(query);
        // Phrase needs positional tokens; boolean AND only needs the term set.
        let analyzed = analyzer::analyze_query(inner, &self.analyzer_config, is_phrase);

        if analyzed.terms.is_empty() {
            return (analyzed, Vec::new(), FastBitmap::new());
        }

        // Take shadow snapshot once for the whole query (applied only to base postings).
        let shadowed = self.shadowed_docs.read();
        let mut postings = Vec::with_capacity(analyzed.terms.len());
        for term in &analyzed.terms {
            let posting = self.search_term_posting(term, &shadowed);
            // AND semantics: any empty term makes the whole result empty.
            if posting.is_empty() {
                return (analyzed, Vec::new(), FastBitmap::new());
            }
            postings.push(posting);
        }
        drop(shadowed);

        let mut bitmap = query::intersect_postings(&postings);

        if !bitmap.is_empty() && is_phrase && self.track_doc_terms {
            let query_tokens = query::map_query_tokens(&analyzed, |term| {
                self.dict.get(term).map(|r| *r.value())
            });
            if !query_tokens.is_empty() {
                let store = self.token_store.read();
                bitmap = query::filter_phrase(&bitmap, &query_tokens, &store);
            }
        }

        // Deleted can be applied once after intersection: (A-D)∩(B-D) = (A∩B)-D.
        if !bitmap.is_empty() {
            let deleted = self.deleted_docs.read();
            if !deleted.is_empty() {
                bitmap.andnot_inplace(&deleted);
            }
        }

        (analyzed, postings, bitmap)
    }
    
    fn search_internal(&self, query: &str) -> FastBitmap {
        self.search_candidates(query).2
    }

    /// Resolve live postings for one term: `(base − shadow) ∪ delta`.
    ///
    /// Does **not** insert into the dictionary (search must be read-only w.r.t. vocab).
    /// Chinese n-gram expansion is handled by the analyzer at query time — each n-gram
    /// is already a separate term in `analyzed.terms` and gets intersected above.
    fn search_term_posting(&self, term: &str, shadowed: &FastBitmap) -> FastBitmap {
        let mut result = match self.index.as_ref().and_then(|index| index.get(term)) {
            Some(mut base) => {
                if !base.is_empty() && !shadowed.is_empty() {
                    base.andnot_inplace(shadowed);
                }
                base
            }
            None => FastBitmap::new(),
        };

        if let Some(term_id) = self.dict.get(term) {
            if let Some(bitmap) = self.buffer.get(&*term_id) {
                result.or_inplace(&bitmap);
            }
        }

        result
    }

    /// Public-style helper used by fuzzy search: posting + deleted filter for a single term.
    fn search_term(&self, term: &str) -> FastBitmap {
        let shadowed = self.shadowed_docs.read();
        let mut result = self.search_term_posting(term, &shadowed);
        drop(shadowed);
        if !result.is_empty() {
            let deleted = self.deleted_docs.read();
            if !deleted.is_empty() {
                result.andnot_inplace(&deleted);
            }
        }
        result
    }
    
    fn fuzzy_search_internal(&self, query: &str) -> FastBitmap {
        let config = self.fuzzy_config.read();
        let threshold = config.threshold;
        let max_distance = config.max_distance;
        let max_candidates = config.max_candidates;
        drop(config);
        
        // Collect unique terms from dictionary + index
        let mut all_terms: Vec<String> = self.dict.iter()
            .map(|e| e.key().clone())
            .collect();
        
        if let Some(ref index) = self.index {
            all_terms.extend(index.all_terms());
        }
        all_terms.sort_unstable();
        all_terms.dedup();
        
        // Find similar terms: length/q-gram prune → early-exit Levenshtein
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut result = FastBitmap::new();
        
        for word in words {
            let pruned = crate::fuzzy::prune_fuzzy_candidates(&all_terms, word, max_distance);
            let mut candidates: Vec<(&str, usize, f64)> = pruned
                .into_iter()
                .filter_map(|t| {
                    let distance = crate::analyzer::levenshtein(word, t, max_distance)?;
                    let width = word.chars().count().max(t.chars().count()).max(1);
                    let similarity = 1.0 - distance as f64 / width as f64;
                    (similarity >= threshold).then_some((t, distance, similarity))
                })
                .collect();
            
            candidates.sort_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            });
            candidates.truncate(max_candidates);
            
            let mut word_result = FastBitmap::new();
            for (term, _, _) in candidates {
                let term_bitmap = self.search_term(term);
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
    max_chinese_length=3,
    min_term_length=2,
    fuzzy_threshold=0.7,
    fuzzy_max_distance=2,
    track_doc_terms=true,
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
