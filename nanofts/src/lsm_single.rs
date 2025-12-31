//! Single-File LSM Index - Simplified High-Performance Version
//!
//! File Layout:
//! ```text
//! [Header 64B]
//! [Data Region]  <- Append writes
//! ```
//!
//! Each flush appends a Segment Block:
//! [block_size: 8B][term_count: 4B][entries...]
//! 
//! Entry format:
//! [term_len: 2B][term][data_len: 4B][compressed_posting]
//!
//! ## WAL Support
//! 
//! Optional WAL (Write-Ahead Log) for crash recovery:
//! - Write operations are first logged to WAL
//! - WAL is cleared after successful flush
//! - Incomplete writes are automatically recovered on startup
//!
//! ## Lazy Load Mode
//!
//! When lazy_load is enabled:
//! - Only index directory is loaded on startup (term -> file offset)
//! - Bitmaps are loaded on demand during search
//! - LRU cache manages memory

use crate::bitmap::FastBitmap;
use crate::vbyte;
use crate::wal::{WriteAheadLog, WalOp};
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"NFS1"; // NanoFTS Single v1
const VERSION: u16 = 1;
const HEADER_SIZE: usize = 64;
const ROARING_THRESHOLD: usize = 128;
const DEFAULT_CACHE_SIZE: usize = 10000; // Default cache 10000 terms

#[derive(Error, Debug)]
pub enum LsmSingleError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
}

/// File header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FileHeader {
    magic: [u8; 4],
    version: u16,
    flags: u16,
    block_count: u64,
    total_terms: u64,
    total_docs: u64,
    _reserved: [u8; 32],
}

/// Term index entry (for lazy load mode)
#[derive(Clone, Debug)]
struct TermIndexEntry {
    /// File offset (pointing to compressed data)
    offset: u64,
    /// Compressed data length
    len: u32,
}

/// In-memory data block
#[allow(dead_code)]
struct DataBlock {
    terms: FxHashMap<String, FastBitmap>,
}

/// Single-file LSM index
pub struct LsmSingleIndex {
    path: PathBuf,
    file: RwLock<File>,
    header: RwLock<FileHeader>,
    /// Loaded data blocks (full load mode)
    data: RwLock<FxHashMap<String, FastBitmap>>,
    /// Term index directory (lazy load mode)
    term_index: RwLock<FxHashMap<String, TermIndexEntry>>,
    /// LRU cache (lazy load mode)
    cache: Mutex<LruCache<String, FastBitmap>>,
    /// Cache hit statistics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    /// Whether lazy load is enabled
    lazy_load: bool,
    /// Write buffer
    buffer: RwLock<FxHashMap<String, FastBitmap>>,
    buffer_size: RwLock<usize>,
    buffer_threshold: usize,
    flushing: AtomicBool,
    /// Whether compacting (block flush during compact)
    compacting: AtomicBool,
    /// WAL (optional)
    wal: Option<WriteAheadLog>,
    wal_enabled: bool,
}

impl LsmSingleIndex {
    /// Create new index (default: WAL enabled, lazy load disabled)
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, LsmSingleError> {
        Self::create_with_options(path, true)
    }
    
    /// Create new index with WAL option
    pub fn create_with_options<P: AsRef<Path>>(path: P, enable_wal: bool) -> Result<Self, LsmSingleError> {
        Self::create_full_options(path, enable_wal, false, DEFAULT_CACHE_SIZE)
    }
    
    /// Create new index with full options
    pub fn create_full_options<P: AsRef<Path>>(
        path: P, 
        enable_wal: bool, 
        lazy_load: bool,
        cache_size: usize,
    ) -> Result<Self, LsmSingleError> {
        let path = path.as_ref().to_path_buf();
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        
        let header = FileHeader {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            block_count: 0,
            total_terms: 0,
            total_docs: 0,
            _reserved: [0; 32],
        };
        
        Self::write_header(&mut file, &header)?;
        file.flush()?;
        
        // Initialize WAL (if enabled)
        let wal = if enable_wal {
            match WriteAheadLog::open(&path) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!("WAL initialization failed: {}, continuing without WAL", e);
                    None
                }
            }
        } else {
            None
        };
        
        let cache_cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        
        Ok(Self {
            path,
            file: RwLock::new(file),
            header: RwLock::new(header),
            data: RwLock::new(FxHashMap::default()),
            term_index: RwLock::new(FxHashMap::default()),
            cache: Mutex::new(LruCache::new(cache_cap)),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            lazy_load,
            buffer: RwLock::new(FxHashMap::default()),
            buffer_size: RwLock::new(0),
            buffer_threshold: 32 * 1024 * 1024,
            flushing: AtomicBool::new(false),
            compacting: AtomicBool::new(false),
            wal,
            wal_enabled: enable_wal,
        })
    }
    
    /// Open existing index (default: WAL enabled, lazy load disabled)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LsmSingleError> {
        Self::open_with_options(path, true)
    }
    
    /// Open existing index with WAL option
    pub fn open_with_options<P: AsRef<Path>>(path: P, enable_wal: bool) -> Result<Self, LsmSingleError> {
        Self::open_full_options(path, enable_wal, false, DEFAULT_CACHE_SIZE)
    }
    
    /// Open existing index with lazy load
    pub fn open_lazy<P: AsRef<Path>>(path: P, cache_size: usize) -> Result<Self, LsmSingleError> {
        Self::open_full_options(path, true, true, cache_size)
    }
    
    /// Open existing index with full options
    pub fn open_full_options<P: AsRef<Path>>(
        path: P, 
        enable_wal: bool,
        lazy_load: bool,
        cache_size: usize,
    ) -> Result<Self, LsmSingleError> {
        let path = path.as_ref().to_path_buf();
        
        if !path.exists() {
            return Self::create_full_options(&path, enable_wal, lazy_load, cache_size);
        }
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        
        let header = Self::read_header(&mut file)?;
        
        if &header.magic != MAGIC {
            return Err(LsmSingleError::InvalidFormat("Invalid magic".into()));
        }
        
        let cache_cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        
        // Load data based on mode
        let (data, term_index) = if lazy_load {
            // Lazy load mode: only load index directory
            let index = Self::load_term_index(&mut file, &header)?;
            (FxHashMap::default(), index)
        } else {
            // Full load mode: load all data
            let data = Self::load_all_blocks(&mut file, &header)?;
            (data, FxHashMap::default())
        };
        
        let mut data = data;
        
        // Initialize WAL and recover incomplete writes
        let wal = if enable_wal {
            match WriteAheadLog::open(&path) {
                Ok(w) => {
                    // Recover data from WAL
                    if let Ok(batches) = w.recover() {
                        let recovered_count: usize = batches.iter().map(|b| b.len()).sum();
                        if recovered_count > 0 {
                            eprintln!("WAL: Recovering {} entries from {} batches", 
                                recovered_count, batches.len());
                            
                            for batch in batches {
                                for entry in batch.entries {
                                    match entry.op {
                                        WalOp::Add => {
                                            data.entry(entry.term)
                                                .or_insert_with(FastBitmap::new)
                                                .add(entry.doc_id);
                                        }
                                        WalOp::Remove => {
                                            if let Some(bitmap) = data.get_mut(&entry.term) {
                                                bitmap.remove(entry.doc_id);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(w)
                }
                Err(e) => {
                    eprintln!("WAL initialization failed: {}, continuing without WAL", e);
                    None
                }
            }
        } else {
            None
        };
        
        Ok(Self {
            path,
            file: RwLock::new(file),
            header: RwLock::new(header),
            data: RwLock::new(data),
            term_index: RwLock::new(term_index),
            cache: Mutex::new(LruCache::new(cache_cap)),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            lazy_load,
            buffer: RwLock::new(FxHashMap::default()),
            buffer_size: RwLock::new(0),
            buffer_threshold: 32 * 1024 * 1024,
            flushing: AtomicBool::new(false),
            compacting: AtomicBool::new(false),
            wal,
            wal_enabled: enable_wal,
        })
    }
    
    fn write_header(file: &mut File, header: &FileHeader) -> Result<(), LsmSingleError> {
        file.seek(SeekFrom::Start(0))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(header as *const FileHeader as *const u8, HEADER_SIZE)
        };
        file.write_all(bytes)?;
        Ok(())
    }
    
    fn read_header(file: &mut File) -> Result<FileHeader, LsmSingleError> {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut bytes)?;
        let header: FileHeader = unsafe { std::ptr::read(bytes.as_ptr() as *const FileHeader) };
        Ok(header)
    }
    
    fn load_all_blocks(file: &mut File, header: &FileHeader) -> Result<FxHashMap<String, FastBitmap>, LsmSingleError> {
        let mut data = FxHashMap::default();
        
        if header.block_count == 0 {
            return Ok(data);
        }
        
        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        
        for _ in 0..header.block_count {
            // Read block size
            let mut size_buf = [0u8; 8];
            if file.read_exact(&mut size_buf).is_err() {
                break;
            }
            let block_size = u64::from_le_bytes(size_buf) as usize;
            
            // Read block data
            let mut block_data = vec![0u8; block_size];
            file.read_exact(&mut block_data)?;
            
            // Parse block
            Self::parse_block(&block_data, &mut data)?;
        }
        
        Ok(data)
    }
    
    /// Load only index directory (lazy load mode)
    /// Returns term -> (file_offset, data_len) mapping
    fn load_term_index(file: &mut File, header: &FileHeader) -> Result<FxHashMap<String, TermIndexEntry>, LsmSingleError> {
        let mut index = FxHashMap::default();
        
        if header.block_count == 0 {
            return Ok(index);
        }
        
        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        
        for _ in 0..header.block_count {
            // Read block size
            let mut size_buf = [0u8; 8];
            if file.read_exact(&mut size_buf).is_err() {
                break;
            }
            let block_size = u64::from_le_bytes(size_buf) as usize;
            let block_start = file.stream_position()?;
            
            // Read block data (parse index only, no decompression)
            let mut block_data = vec![0u8; block_size];
            file.read_exact(&mut block_data)?;
            
            // Parse index
            Self::parse_block_index(&block_data, block_start, &mut index)?;
        }
        
        Ok(index)
    }
    
    /// Parse block's index directory (no data decompression)
    fn parse_block_index(
        data: &[u8], 
        block_file_offset: u64,
        result: &mut FxHashMap<String, TermIndexEntry>
    ) -> Result<(), LsmSingleError> {
        if data.len() < 4 {
            return Ok(());
        }
        
        let term_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        
        for _ in 0..term_count {
            if offset + 2 > data.len() {
                break;
            }
            
            // term_len
            let term_len = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap()) as usize;
            offset += 2;
            
            if offset + term_len > data.len() {
                break;
            }
            
            // term
            let term = String::from_utf8_lossy(&data[offset..offset+term_len]).to_string();
            offset += term_len;
            
            if offset + 4 > data.len() {
                break;
            }
            
            // data_len
            let data_len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap());
            offset += 4;
            
            // Record file offset of compressed data
            let file_offset = block_file_offset + offset as u64;
            
            // Skip compressed data
            offset += data_len as usize;
            
            // Save index entry (keep latest if duplicate term)
            result.insert(term, TermIndexEntry {
                offset: file_offset,
                len: data_len,
            });
        }
        
        Ok(())
    }
    
    fn parse_block(data: &[u8], result: &mut FxHashMap<String, FastBitmap>) -> Result<(), LsmSingleError> {
        if data.len() < 4 {
            return Ok(());
        }
        
        let term_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        
        for _ in 0..term_count {
            if offset + 2 > data.len() {
                break;
            }
            
            // term_len
            let term_len = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap()) as usize;
            offset += 2;
            
            if offset + term_len > data.len() {
                break;
            }
            
            // term
            let term = String::from_utf8_lossy(&data[offset..offset+term_len]).to_string();
            offset += term_len;
            
            if offset + 4 > data.len() {
                break;
            }
            
            // data_len
            let data_len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            
            if offset + data_len > data.len() {
                break;
            }
            
            // compressed posting
            let compressed = &data[offset..offset+data_len];
            offset += data_len;
            
            // Decompress
            if let Ok(decompressed) = zstd::decode_all(compressed) {
                // Decode bitmap
                let bitmap = if let Ok(b) = FastBitmap::deserialize(&decompressed) {
                    b
                } else if let Some((ids, _)) = vbyte::decode_sorted_u32_array(&decompressed) {
                    FastBitmap::from_iter(ids)
                } else {
                    continue;
                };
                
                // Merge into result
                result.entry(term)
                    .and_modify(|existing| existing.or_inplace(&bitmap))
                    .or_insert(bitmap);
            }
        }
        
        Ok(())
    }
    
    /// Load single term's bitmap from file on demand
    fn load_term_from_file(&self, entry: &TermIndexEntry) -> Option<FastBitmap> {
        let mut file = self.file.write();
        
        // Seek to compressed data position
        if file.seek(SeekFrom::Start(entry.offset)).is_err() {
            return None;
        }
        
        // Read compressed data
        let mut compressed = vec![0u8; entry.len as usize];
        if file.read_exact(&mut compressed).is_err() {
            return None;
        }
        
        // Decompress
        let decompressed = match zstd::decode_all(&compressed[..]) {
            Ok(d) => d,
            Err(_) => return None,
        };
        
        // Decode bitmap
        if let Ok(bitmap) = FastBitmap::deserialize(&decompressed) {
            Some(bitmap)
        } else if let Some((ids, _)) = vbyte::decode_sorted_u32_array(&decompressed) {
            Some(FastBitmap::from_iter(ids))
        } else {
            None
        }
    }
    
    /// Insert single term-doc pair
    #[inline]
    pub fn upsert(&self, term: &str, doc_id: u64) {
        // Write to WAL first (if enabled)
        if let Some(ref wal) = self.wal {
            wal.log_add(term, doc_id as u32);
        }
        
        let mut buffer = self.buffer.write();
        let is_new = !buffer.contains_key(term);
        
        buffer.entry(term.to_string())
            .or_insert_with(FastBitmap::new)
            .add(doc_id as u32);
        
        drop(buffer);
        
        let mut size = self.buffer_size.write();
        if is_new {
            *size += 64 + term.len();
        }
        *size += 3;
        
        let current_size = *size;
        drop(size);
        
        if current_size >= self.buffer_threshold {
            self.maybe_flush();
        }
    }
    
    /// Batch insert term-bitmap pairs (for flushing from UnifiedEngine's buffer)
    pub fn upsert_batch(&self, entries: Vec<(String, FastBitmap)>) {
        // Batch write to WAL (if enabled)
        if let Some(ref wal) = self.wal {
            for (term, bitmap) in &entries {
                for doc_id in bitmap.iter() {
                    wal.log_add(term, doc_id);
                }
            }
            // Commit WAL batch
            if let Err(e) = wal.commit() {
                eprintln!("WAL commit failed: {}", e);
            }
        }
        
        let mut buffer = self.buffer.write();
        let mut added_size = 0;
        
        for (term, bitmap) in entries {
            let is_new = !buffer.contains_key(&term);
            let bitmap_len = bitmap.len() as usize;
            
            buffer.entry(term.clone())
                .and_modify(|existing| existing.or_inplace(&bitmap))
                .or_insert(bitmap);
            
            if is_new {
                added_size += 64 + term.len();
            }
            added_size += bitmap_len * 3;
        }
        
        drop(buffer);
        
        *self.buffer_size.write() += added_size;
        
        if *self.buffer_size.read() >= self.buffer_threshold {
            self.maybe_flush();
        }
    }
    
    /// Commit current WAL batch
    pub fn commit_wal(&self) -> Result<usize, LsmSingleError> {
        if let Some(ref wal) = self.wal {
            wal.commit().map_err(|e| LsmSingleError::IoError(
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            ))
        } else {
            Ok(0)
        }
    }
    
    pub fn get(&self, term: &str) -> Option<FastBitmap> {
        // 1. Query buffer (latest data)
        let buffer = self.buffer.read();
        let buf_result = buffer.get(term).cloned();
        drop(buffer);
        
        // 2. Query persisted data based on mode
        let persisted_result = if self.lazy_load {
            self.get_lazy(term)
        } else {
            // Full load mode: read directly from memory
            let data = self.data.read();
            data.get(term).cloned()
        };
        
        // 3. Merge results
        match (buf_result, persisted_result) {
            (Some(b), Some(d)) => Some(b.or(&d)),
            (Some(b), None) => Some(b),
            (None, Some(d)) => Some(d),
            (None, None) => None,
        }
    }
    
    /// Lazy load mode query
    fn get_lazy(&self, term: &str) -> Option<FastBitmap> {
        // 1. Check LRU cache first
        {
            let mut cache = self.cache.lock();
            if let Some(bitmap) = cache.get(term) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(bitmap.clone());
            }
        }
        
        // 2. Cache miss, look up in index directory
        let entry = {
            let index = self.term_index.read();
            index.get(term).cloned()
        };
        
        let entry = match entry {
            Some(e) => e,
            None => {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        
        // 3. Load from file
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let bitmap = self.load_term_from_file(&entry)?;
        
        // 4. Put into cache
        {
            let mut cache = self.cache.lock();
            cache.put(term.to_string(), bitmap.clone());
        }
        
        Some(bitmap)
    }
    
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64, usize) {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let cache_len = self.cache.lock().len();
        (hits, misses, cache_len)
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
    
    /// Whether lazy load is enabled
    pub fn is_lazy_load(&self) -> bool {
        self.lazy_load
    }
    
    /// Warmup cache (load specified terms)
    pub fn warmup(&self, terms: &[String]) -> usize {
        if !self.lazy_load {
            return 0;
        }
        
        let mut loaded = 0;
        for term in terms {
            if self.get(term).is_some() {
                loaded += 1;
            }
        }
        loaded
    }
    
    fn maybe_flush(&self) {
        // Skip auto flush if compacting
        if self.compacting.load(Ordering::SeqCst) {
            return;
        }
        if self.flushing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            let _ = self.flush_internal();
            self.flushing.store(false, Ordering::SeqCst);
        }
    }
    
    fn flush_internal(&self) -> Result<(), LsmSingleError> {
        let entries: Vec<_> = self.buffer.write().drain().collect();
        if entries.is_empty() {
            // Even if buffer is empty, clear WAL (may have uncommitted entries)
            if let Some(ref wal) = self.wal {
                let _ = wal.clear();
            }
            return Ok(());
        }
        
        *self.buffer_size.write() = 0;
        
        // Build block data
        let mut block = Vec::new();
        
        // term_count
        block.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        
        for (term, bitmap) in &entries {
            // term_len + term
            block.extend_from_slice(&(term.len() as u16).to_le_bytes());
            block.extend_from_slice(term.as_bytes());
            
            // Serialize bitmap
            let serialized = if bitmap.len() < ROARING_THRESHOLD as u64 {
                let ids: Vec<u32> = bitmap.to_vec();
                let mut buf = Vec::new();
                vbyte::encode_sorted_u32_array(&ids, &mut buf);
                buf
            } else {
                bitmap.serialize().unwrap_or_default()
            };
            
            // Compress
            let compressed = zstd::encode_all(&serialized[..], 1)
                .map_err(|e| LsmSingleError::CompressionError(e.to_string()))?;
            
            // data_len + data
            block.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            block.extend(compressed);
        }
        
        // Write to file
        let mut file = self.file.write();
        file.seek(SeekFrom::End(0))?;
        
        // block_size + block
        file.write_all(&(block.len() as u64).to_le_bytes())?;
        file.write_all(&block)?;
        
        // Update header
        let mut header = self.header.write();
        header.block_count += 1;
        header.total_terms += entries.len() as u64;
        header.total_docs += entries.iter().map(|(_, b)| b.len()).sum::<u64>();
        
        Self::write_header(&mut file, &header)?;
        file.sync_all()?; // Ensure data is written to disk
        
        drop(header);
        drop(file);
        
        // Update in-memory data based on mode
        if self.lazy_load {
            // Lazy load mode: update index directory and put into cache
            // Rescan file to get new offsets
            let mut file = self.file.write();
            let header = self.header.read();
            if let Ok(new_index) = Self::load_term_index(&mut file, &header) {
                drop(header);
                drop(file);
                
                // Merge new index
                let mut term_index = self.term_index.write();
                for (term, entry) in new_index {
                    term_index.insert(term, entry);
                }
            }
            
            // Put newly written data into cache
            // Important: merge with old cache data, don't overwrite
            let mut cache = self.cache.lock();
            for (term, bitmap) in entries {
                if let Some(existing) = cache.get(&term) {
                    let merged = existing.or(&bitmap);
                    cache.put(term, merged);
                } else {
                    cache.put(term, bitmap);
                }
            }
        } else {
            // Full load mode: update in-memory data
            let mut data = self.data.write();
            for (term, bitmap) in entries {
                data.entry(term)
                    .and_modify(|existing| existing.or_inplace(&bitmap))
                    .or_insert(bitmap);
            }
        }
        
        // Data persisted, clear WAL
        if let Some(ref wal) = self.wal {
            if let Err(e) = wal.clear() {
                eprintln!("WAL clear failed: {}", e);
            }
        }
        
        Ok(())
    }
    
    pub fn flush(&self) -> Result<(), LsmSingleError> {
        // Wait for compact to complete
        while self.compacting.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        while self.flushing.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        self.flushing.store(true, Ordering::SeqCst);
        let result = self.flush_internal();
        self.flushing.store(false, Ordering::SeqCst);
        result
    }
    
    pub fn compact(&self) -> Result<(), LsmSingleError> {
        self.compact_with_deletions(&[])
    }
    
    /// Compact and apply deletions
    pub fn compact_with_deletions(&self, deleted_docs: &[u32]) -> Result<(), LsmSingleError> {
        // Set compacting flag to block other flush operations
        self.compacting.store(true, Ordering::SeqCst);
        
        // Ensure flag is cleared on function return
        struct CompactGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for CompactGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = CompactGuard(&self.compacting);
        
        self.flush_internal()?;
        
        // Create temp file
        let tmp_path = self.path.with_extension("nfts.tmp");
        
        {
            // Disable WAL for temp index as this is atomic operation, no recovery needed
            let new_index = Self::create_with_options(&tmp_path, false)?;
            
            // Get all terms based on mode
            if self.lazy_load {
                // Lazy load mode: need to get data from index directory and cache
                let term_index = self.term_index.read();
                for (term, entry) in term_index.iter() {
                    // Load bitmap
                    if let Some(mut bitmap) = self.load_term_from_file(entry) {
                        // Apply deletions
                        for &doc_id in deleted_docs {
                            bitmap.remove(doc_id);
                        }
                        
                        // Only write non-empty bitmaps
                        if !bitmap.is_empty() {
                            new_index.buffer.write().insert(term.clone(), bitmap);
                        }
                    }
                }
            } else {
                // Full load mode: copy from in-memory data
                let data = self.data.read();
                for (term, bitmap) in data.iter() {
                    let mut new_bitmap = bitmap.clone();
                    
                    // Apply deletions
                    for &doc_id in deleted_docs {
                        new_bitmap.remove(doc_id);
                    }
                    
                    // Only write non-empty bitmaps
                    if !new_bitmap.is_empty() {
                        new_index.buffer.write().insert(term.clone(), new_bitmap);
                    }
                }
            }
            new_index.flush()?;
        }
        
        // Replace file
        std::fs::rename(&tmp_path, &self.path)?;
        
        // Clean up potentially remaining temp WAL file
        let tmp_wal_path = tmp_path.with_extension("nfts.wal");
        if tmp_wal_path.exists() {
            let _ = std::fs::remove_file(&tmp_wal_path);
        }
        // Also clean up incorrectly named WAL file (index.nfts.nfts.wal)
        let wrong_wal_path = self.path.with_extension("nfts.nfts.wal");
        if wrong_wal_path.exists() {
            let _ = std::fs::remove_file(&wrong_wal_path);
        }
        
        // Reopen
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        
        let new_header = Self::read_header(&mut file)?;
        
        // Reload based on mode
        if self.lazy_load {
            let new_index = Self::load_term_index(&mut file, &new_header)?;
            *self.term_index.write() = new_index;
            self.clear_cache();
        } else {
            let new_data = Self::load_all_blocks(&mut file, &new_header)?;
            *self.data.write() = new_data;
        }
        
        *self.file.write() = file;
        *self.header.write() = new_header;
        
        Ok(())
    }
    
    pub fn term_count(&self) -> usize {
        let persisted_count = if self.lazy_load {
            self.term_index.read().len()
        } else {
            self.data.read().len()
        };
        persisted_count + self.buffer.read().len()
    }
    
    pub fn segment_count(&self) -> usize {
        self.header.read().block_count as usize
    }
    
    pub fn memtable_size(&self) -> usize {
        *self.buffer_size.read()
    }
    
    /// Get WAL file size
    pub fn wal_size(&self) -> u64 {
        self.wal.as_ref()
            .and_then(|w| w.file_size().ok())
            .unwrap_or(0)
    }
    
    /// Get WAL pending batch count
    pub fn wal_pending_batches(&self) -> u64 {
        self.wal.as_ref()
            .map(|w| w.pending_batch_count())
            .unwrap_or(0)
    }
    
    /// Check if WAL is enabled
    pub fn is_wal_enabled(&self) -> bool {
        self.wal_enabled && self.wal.is_some()
    }
    
    pub fn all_terms(&self) -> Vec<String> {
        let mut terms: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        // Get persisted terms based on mode
        if self.lazy_load {
            terms.extend(self.term_index.read().keys().cloned());
        } else {
            terms.extend(self.data.read().keys().cloned());
        }
        
        // Add buffer terms
        terms.extend(self.buffer.read().keys().cloned());
        terms.into_iter().collect()
    }
    
    /// Remove specified document IDs from all data
    pub fn remove_docs(&self, doc_ids: &[u32]) {
        // Remove from buffer
        for entry in self.buffer.write().values_mut() {
            for &doc_id in doc_ids {
                entry.remove(doc_id);
            }
        }
        
        // Remove from loaded data
        for entry in self.data.write().values_mut() {
            for &doc_id in doc_ids {
                entry.remove(doc_id);
            }
        }
    }
}

impl Drop for LsmSingleIndex {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
