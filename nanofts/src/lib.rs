//! NanoFTS Core - Ultra High-Performance Full-Text Search Engine Core
//! 
//! Optimized for billion-scale data with sub-millisecond search response
//! 
//! # Main Features
//! - LSM-Tree Architecture: No scale limits
//! - Incremental Writes: Real-time updates
//! - Fuzzy Search
//! - Zero-copy Result Handle
//! - Result Set Operations (AND/OR/NOT)
//! - NumPy Return Support

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

pub use bitmap::*;
pub use cache::*;
pub use index::*;
pub use search::*;
pub use shard::*;

use pyo3::prelude::*;

/// NanoFTS Python Module
#[pymodule]
fn nanofts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Unified engine
    m.add_class::<unified_engine::UnifiedEngine>()?;
    m.add_class::<unified_engine::ResultHandle>()?;
    m.add_class::<unified_engine::FuzzyConfig>()?;
    
    // Main API
    m.add_function(wrap_pyfunction!(unified_engine::create_engine, m)?)?;
    
    // Aliases
    m.add("SearchEngine", m.getattr("UnifiedEngine")?)?;
    m.add("SearchResult", m.getattr("ResultHandle")?)?;
    
    Ok(())
}
