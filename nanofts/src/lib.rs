//! NanoFTS - Ultra High-Performance Full-Text Search Engine
//!
//! A high-performance full-text search engine written in Rust, optimized for
//! billion-scale data with sub-millisecond search response.
//!
//! # Main Features
//!
//! - **LSM-Tree Architecture**: No scale limits with efficient disk-based storage
//! - **Incremental Writes**: Real-time updates with WAL support
//! - **Fuzzy Search**: Similarity-based search with configurable thresholds
//! - **Zero-copy Result Handle**: Efficient memory usage with shared bitmap results
//! - **Result Set Operations**: AND/OR/NOT operations on search results
//! - **Lazy Load Mode**: Memory-efficient loading for large indexes
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use nanofts::{UnifiedEngine, EngineConfig};
//! use std::collections::HashMap;
//!
//! // Create an in-memory search engine
//! let engine = UnifiedEngine::new(EngineConfig::memory_only()).unwrap();
//!
//! // Add a document
//! let mut fields = HashMap::new();
//! fields.insert("title".to_string(), "Hello World".to_string());
//! fields.insert("content".to_string(), "This is a test document".to_string());
//! engine.add_document(1, fields).unwrap();
//!
//! // Search
//! let result = engine.search("hello").unwrap();
//! println!("Found {} documents", result.total_hits());
//!
//! // Get document IDs
//! for doc_id in result.iter() {
//!     println!("Document ID: {}", doc_id);
//! }
//! ```
//!
//! # Persistent Storage
//!
//! ```rust,no_run
//! use nanofts::{UnifiedEngine, EngineConfig};
//!
//! // Create a persistent search engine
//! let config = EngineConfig::persistent("my_index.nfts")
//!     .with_lazy_load(true)       // Enable lazy loading for large indexes
//!     .with_cache_size(10000);    // LRU cache size
//!
//! let engine = UnifiedEngine::new(config).unwrap();
//!
//! // ... add documents and search ...
//!
//! // Flush to disk
//! engine.flush().unwrap();
//! ```
//!
//! # Features
//!
//! - `python` - Enable Python bindings via PyO3 (disabled by default)
//! - `simd` - Enable SIMD acceleration (requires nightly)
//! - `mimalloc` - Use mimalloc allocator (enabled by default)

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Core modules
pub mod bitmap;
pub mod cache;
pub mod index;
pub mod search;
pub mod shard;
pub mod simd_utils;
pub mod vbyte;
pub mod wal;
pub mod lsm_single;
pub mod unified_engine;

// Re-export core types
pub use bitmap::{FastBitmap, BitmapError, fast_intersection, fast_union};
pub use cache::*;
pub use index::*;
pub use search::*;
pub use shard::*;
pub use lsm_single::{LsmSingleIndex, LsmSingleError};
pub use unified_engine::{
    UnifiedEngine,
    ResultHandle,
    FuzzyConfig,
    EngineConfig,
    EngineError,
    EngineResult,
    create_engine,
};

// Python module (only when python feature is enabled)
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// NanoFTS Python Module
#[cfg(feature = "python")]
#[pymodule]
fn nanofts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Unified engine
    m.add_class::<unified_engine::UnifiedEngine>()?;
    m.add_class::<unified_engine::ResultHandle>()?;
    m.add_class::<unified_engine::FuzzyConfig>()?;
    
    // Main API
    m.add_function(wrap_pyfunction!(unified_engine::create_engine_py, m)?)?;
    
    // Aliases
    m.add("SearchEngine", m.getattr("UnifiedEngine")?)?;
    m.add("SearchResult", m.getattr("ResultHandle")?)?;
    m.add("create_engine", m.getattr("create_engine_py")?)?;
    
    Ok(())
}
