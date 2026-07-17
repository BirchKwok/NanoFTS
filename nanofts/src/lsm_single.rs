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
use crate::term_dir::TermDirectory;
use crate::vbyte;
use crate::wal::{WriteAheadLog, WalOp};
use dashmap::DashMap;
use fork_union::spawn;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"NFS2"; // NanoFTS Single v2 (u64 doc ids)
const LEGACY_MAGIC: &[u8; 4] = b"NFS1"; // NanoFTS Single v1 (u32 doc ids, unsupported)
const VERSION: u16 = 2;
const ANALYZER_VERSION: u32 = 2;
const HEADER_SIZE: usize = 64;
const ROARING_THRESHOLD: usize = 128;
const DEFAULT_CACHE_SIZE: usize = 10000; // Default cache 10000 terms
/// Uncompressed posting codecs (before zstd).
const POSTING_CODEC_VBYTE: u8 = 0;
const POSTING_CODEC_SINGLE: u8 = 1;
const POSTING_CODEC_ROARING: u8 = 2;

#[derive(Error, Debug)]
pub enum LsmSingleError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("Checksum mismatch")]
    ChecksumMismatch,
}

/// File header (on-disk layout is little-endian; see encode/decode)
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

fn encode_posting(bitmap: &FastBitmap) -> Result<Vec<u8>, LsmSingleError> {
    let mut serialized = Vec::new();
    if bitmap.len() == 1 {
        serialized.push(POSTING_CODEC_SINGLE);
        let id = bitmap.iter().next().unwrap();
        serialized.extend_from_slice(&id.to_le_bytes());
    } else if bitmap.len() < ROARING_THRESHOLD as u64 {
        serialized.push(POSTING_CODEC_VBYTE);
        let ids: Vec<u64> = bitmap.to_vec();
        vbyte::encode_sorted_u64_array(&ids, &mut serialized);
    } else {
        serialized.push(POSTING_CODEC_ROARING);
        let roaring = bitmap
            .serialize()
            .map_err(|e| LsmSingleError::CompressionError(e.to_string()))?;
        serialized.extend_from_slice(&roaring);
    }
    zstd::encode_all(&serialized[..], 1).map_err(|e| LsmSingleError::CompressionError(e.to_string()))
}

fn decode_posting(compressed: &[u8]) -> Option<FastBitmap> {
    let decompressed = zstd::decode_all(compressed).ok()?;
    if decompressed.is_empty() {
        return None;
    }
    // NFS2 codec-prefixed payloads
    match decompressed[0] {
        POSTING_CODEC_SINGLE if decompressed.len() == 9 => {
            let id = u64::from_le_bytes(decompressed[1..9].try_into().ok()?);
            Some(FastBitmap::from_iter([id]))
        }
        POSTING_CODEC_VBYTE => {
            let (ids, _) = vbyte::decode_sorted_u64_array(&decompressed[1..])?;
            Some(FastBitmap::from_iter(ids))
        }
        POSTING_CODEC_ROARING => FastBitmap::deserialize(&decompressed[1..]).ok(),
        // Legacy unprefixed payloads (same process, pre-codec): try roaring then vbyte
        _ => {
            if let Ok(b) = FastBitmap::deserialize(&decompressed) {
                Some(b)
            } else if let Some((ids, _)) = vbyte::decode_sorted_u64_array(&decompressed) {
                Some(FastBitmap::from_iter(ids))
            } else {
                None
            }
        }
    }
}

impl FileHeader {
    /// Encode to a portable little-endian 64-byte buffer.
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.block_count.to_le_bytes());
        buf[16..24].copy_from_slice(&self.total_terms.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_docs.to_le_bytes());
        buf[32..64].copy_from_slice(&self._reserved);
        buf
    }

    /// Store the analyzer version as u32 LE in the first 4 bytes of `_reserved`.
    fn set_analyzer_version(&mut self, analyzer_version: u32) {
        self._reserved[0..4].copy_from_slice(&analyzer_version.to_le_bytes());
    }

    /// Read the analyzer version stored in the first 4 bytes of `_reserved`.
    #[allow(dead_code)]
    fn analyzer_version(&self) -> u32 {
        u32::from_le_bytes(self._reserved[0..4].try_into().unwrap())
    }

    /// Store CRC32 of payload (bytes after header) in reserved[20..24].
    fn set_payload_crc(&mut self, crc: u32) {
        self._reserved[20..24].copy_from_slice(&crc.to_le_bytes());
    }

    fn payload_crc(&self) -> u32 {
        u32::from_le_bytes(self._reserved[20..24].try_into().unwrap())
    }

    /// Decode from a portable little-endian 64-byte buffer.
    fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        Self {
            magic: buf[0..4].try_into().unwrap(),
            version: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            flags: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
            block_count: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            total_terms: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            total_docs: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            _reserved: buf[32..64].try_into().unwrap(),
        }
    }
}

/// Term index entry (for lazy load mode)
/// Stores multiple locations since a term may appear in multiple blocks
#[derive(Clone, Debug)]
struct TermIndexEntry {
    /// List of (file_offset, data_len) pairs
    /// Each pair points to compressed data in a different block
    locations: Vec<(u64, u32)>,
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
    /// Loaded data blocks (full load mode) — DashMap for lock-free parallel reads/writes
    data: DashMap<String, FastBitmap>,
    /// Term index directory (lazy load mode)
    term_index: RwLock<FxHashMap<String, TermIndexEntry>>,
    /// Sorted, mmap-friendly term directory sidecar (lazy load fast path).
    /// Built by `compact_with_deletions` from the fully-compacted live postings;
    /// checked before falling back to `term_index` in `get_lazy`.
    term_dir: RwLock<Option<TermDirectory>>,
    /// Set when a persisted `.tdir` sidecar exists but could not be loaded
    /// (missing/corrupt/stale). A `compact()` call rebuilds it.
    needs_rebuild: AtomicBool,
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
        
        let mut header = FileHeader {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            block_count: 0,
            total_terms: 0,
            total_docs: 0,
            _reserved: [0; 32],
        };
        header.set_analyzer_version(ANALYZER_VERSION);
        
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
            data: DashMap::new(),
            term_index: RwLock::new(FxHashMap::default()),
            term_dir: RwLock::new(None),
            needs_rebuild: AtomicBool::new(false),
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
        
        if &header.magic == LEGACY_MAGIC {
            drop(file);
            let quarantine_path = {
                let mut p = path.clone().into_os_string();
                p.push(".incompatible");
                PathBuf::from(p)
            };
            let _ = std::fs::rename(&path, &quarantine_path);
            return Err(LsmSingleError::InvalidFormat(
                "Legacy NFS1 format is not supported; rebuild the index".into(),
            ));
        }
        
        if &header.magic != MAGIC {
            return Err(LsmSingleError::InvalidFormat("Invalid magic".into()));
        }

        // Verify payload CRC when present (skip in lazy_load for fast open of huge files).
        if !lazy_load && header.payload_crc() != 0 {
            match Self::compute_payload_crc(&mut file) {
                Ok(crc) if crc == header.payload_crc() => {}
                Ok(_) | Err(_) => {
                    drop(file);
                    let quarantine_path = {
                        let mut p = path.clone().into_os_string();
                        p.push(".incompatible");
                        PathBuf::from(p)
                    };
                    let _ = std::fs::rename(&path, &quarantine_path);
                    return Err(LsmSingleError::ChecksumMismatch);
                }
            }
        }
        
        let cache_cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        
        // Load data based on mode
        let (data_map, term_index) = if lazy_load {
            // Lazy load mode: only load index directory
            let index = Self::load_term_index(&mut file, &header)?;
            (FxHashMap::default(), index)
        } else {
            // Full load mode: load all data
            let data = Self::load_all_blocks(&mut file, &header)?;
            (data, FxHashMap::default())
        };
        
        // Opportunistically load the sorted mmap term directory sidecar (lazy load
        // fast path only). Missing sidecar is normal (never compacted yet with this
        // feature); an existing-but-unreadable sidecar just falls back to `term_index`
        // and flags `needs_rebuild` so callers can decide to `compact()`.
        let (term_dir_loaded, tdir_needs_rebuild) = if lazy_load {
            match Self::load_term_dir_sidecar(&path) {
                Ok(Some(dir)) => (Some(dir), false),
                Ok(None) => (None, false),
                Err(e) => {
                    eprintln!(
                        "Term directory sidecar for {:?} is invalid ({e}); falling back to full index scan; run compact() to rebuild it",
                        path
                    );
                    (None, true)
                }
            }
        } else {
            (None, false)
        };
        
        let mut data_map = data_map;
        
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
                                            data_map.entry(entry.term)
                                                .or_insert_with(FastBitmap::new)
                                                .add(entry.doc_id);
                                        }
                                        WalOp::Remove => {
                                            if let Some(bitmap) = data_map.get_mut(&entry.term) {
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
        
        let data: DashMap<String, FastBitmap> = data_map.into_iter().collect();

        Ok(Self {
            path,
            file: RwLock::new(file),
            header: RwLock::new(header),
            data,
            term_index: RwLock::new(term_index),
            term_dir: RwLock::new(term_dir_loaded),
            needs_rebuild: AtomicBool::new(tdir_needs_rebuild),
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
        file.write_all(&header.to_bytes())?;
        Ok(())
    }
    
    fn read_header(file: &mut File) -> Result<FileHeader, LsmSingleError> {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut bytes)?;
        Ok(FileHeader::from_bytes(&bytes))
    }

    /// Path of the sorted term directory sidecar next to the main index file.
    fn tdir_path_for(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_os_string();
        p.push(".tdir");
        PathBuf::from(p)
    }

    fn tdir_path(&self) -> PathBuf {
        Self::tdir_path_for(&self.path)
    }

    /// Load the `.tdir` sidecar next to `path`, if present.
    /// `Ok(None)` means no sidecar exists yet (not an error); `Err` means a
    /// sidecar exists but failed to parse (corrupt/stale format).
    fn load_term_dir_sidecar(path: &Path) -> Result<Option<TermDirectory>, std::io::Error> {
        let tdir_path = Self::tdir_path_for(path);
        if !tdir_path.exists() {
            return Ok(None);
        }
        TermDirectory::open(&tdir_path).map(Some)
    }

    /// (Re)build the sorted term directory sidecar from the fully-compacted
    /// live postings and load it as the new lazy-load fast path.
    /// On any failure, leaves `term_dir` cleared and sets `needs_rebuild`.
    fn write_term_dir_sidecar(&self, live_entries: &[(String, FastBitmap)]) {
        let build = || -> Result<TermDirectory, LsmSingleError> {
            let mut sorted: Vec<(String, Vec<u8>)> = Vec::with_capacity(live_entries.len());
            for (term, bitmap) in live_entries {
                let compressed = encode_posting(bitmap)?;
                sorted.push((term.clone(), compressed));
            }
            sorted.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            crate::term_dir::write_term_dir(self.tdir_path(), &sorted)?;
            TermDirectory::open(self.tdir_path()).map_err(LsmSingleError::IoError)
        };

        match build() {
            Ok(dir) => {
                *self.term_dir.write() = Some(dir);
                self.needs_rebuild.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                eprintln!("Failed to (re)build term directory sidecar: {e}");
                *self.term_dir.write() = None;
                self.needs_rebuild.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Fast-path lookup against the sorted mmap term directory, if loaded.
    fn get_from_term_dir(&self, term: &str) -> Option<FastBitmap> {
        let guard = self.term_dir.read();
        let dir = guard.as_ref()?;
        let compressed = dir.get(term)?;
        decode_posting(compressed)
    }

    /// CRC32 over all bytes after the 64-byte header.
    fn compute_payload_crc(file: &mut File) -> Result<u32, LsmSingleError> {
        use crc32fast::Hasher;
        let len = file.metadata()?.len();
        if len <= HEADER_SIZE as u64 {
            return Ok(0);
        }
        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        let mut hasher = Hasher::new();
        let mut buf = [0u8; 64 * 1024];
        let mut remaining = (len - HEADER_SIZE as u64) as usize;
        while remaining > 0 {
            let n = remaining.min(buf.len());
            file.read_exact(&mut buf[..n])?;
            hasher.update(&buf[..n]);
            remaining -= n;
        }
        Ok(hasher.finalize())
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
            
            // Save index entry (append if duplicate term - a term may appear in multiple blocks)
            result.entry(term)
                .or_insert_with(|| TermIndexEntry { locations: Vec::new() })
                .locations.push((file_offset, data_len));
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
            
            if let Some(bitmap) = decode_posting(compressed) {
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
        let mut result: Option<FastBitmap> = None;
        
        // Load and merge all locations for this term
        for &(offset, len) in &entry.locations {
            // Seek to compressed data position
            if file.seek(SeekFrom::Start(offset)).is_err() {
                continue;
            }
            
            // Read compressed data
            let mut compressed = vec![0u8; len as usize];
            if file.read_exact(&mut compressed).is_err() {
                continue;
            }
            
            let bitmap = match decode_posting(&compressed) {
                Some(b) => b,
                None => continue,
            };
            
            // Merge into result
            result = match result {
                Some(existing) => Some(existing.or(&bitmap)),
                None => Some(bitmap),
            };
        }
        
        result
    }
    
    /// Insert single term-doc pair
    #[inline]
    pub fn upsert(&self, term: &str, doc_id: u64) {
        // Write to WAL first (if enabled)
        if let Some(ref wal) = self.wal {
            wal.log_add(term, doc_id);
        }
        
        let mut buffer = self.buffer.write();
        let is_new = !buffer.contains_key(term);
        
        buffer.entry(term.to_string())
            .or_insert_with(FastBitmap::new)
            .add(doc_id);
        
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
        // Batch write to WAL (if enabled) — collect all (term, doc_id) pairs
        // and submit in one lock acquisition instead of per-pair locking
        if let Some(ref wal) = self.wal {
            let mut wal_entries: Vec<(String, u64)> = Vec::new();
            for (term, bitmap) in &entries {
                for doc_id in bitmap.iter() {
                    wal_entries.push((term.clone(), doc_id));
                }
            }
            if !wal_entries.is_empty() {
                wal.log_add_batch(&wal_entries);
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

    /// Merge entries directly into in-memory `data` for immediate searchability.
    /// No WAL write, no buffer write, no disk I/O.
    /// Only effective in full-load mode (lazy_load=false); a no-op otherwise.
    /// Intended for `flush_async`: make data searchable on the calling thread
    /// before handing off the actual disk write to a background thread.
    /// Uses fork_union parallel inserts for large batches.
    pub fn merge_into_data(&self, entries: &[(String, FastBitmap)]) {
        if self.lazy_load || entries.is_empty() {
            return;
        }
        let n = entries.len();
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(n)
            .max(1);
        let mut pool = spawn(num_threads);
        pool.for_n(n, |prong| {
            let (term, bitmap) = &entries[prong.task_index];
            self.data.entry(term.clone())
                .and_modify(|existing| existing.or_inplace(bitmap))
                .or_insert_with(|| bitmap.clone());
        });
    }

    /// Write entries into the write buffer without WAL and without triggering
    /// the auto-flush threshold check. The caller must subsequently call
    /// `flush()` to persist the data.
    /// Intended for the background thread spawned by `flush_async`.
    pub fn enqueue_for_flush(&self, entries: Vec<(String, FastBitmap)>) {
        let mut buffer = self.buffer.write();
        let mut added_size = 0usize;
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
            // Full load mode: read directly from DashMap (no global lock)
            self.data.get(term).map(|r| r.value().clone())
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
        
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // 2. Fast path: sorted mmap term directory built at the last compact.
        // Binary search directly on the mapped file, no per-term String
        // allocation or full-index scan required.
        if let Some(bitmap) = self.get_from_term_dir(term) {
            let mut cache = self.cache.lock();
            cache.put(term.to_string(), bitmap.clone());
            return Some(bitmap);
        }
        
        // 3. Fallback: legacy HashMap directory. Covers terms written since the
        // last compact (not yet reflected in the directory sidecar) and indexes
        // that don't have a sidecar at all.
        let entry = {
            let index = self.term_index.read();
            index.get(term).cloned()
        };
        let entry = entry?;
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
    
    /// True when the persisted term directory sidecar (`.tdir`) is missing,
    /// stale, or corrupt and a `compact()` call is recommended to rebuild it
    /// and restore the lazy-load mmap fast path.
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild.load(Ordering::Relaxed)
    }
    
    /// True if the sorted mmap term directory (lazy-load fast path) is
    /// currently loaded and being consulted by `get()`.
    pub fn term_dir_loaded(&self) -> bool {
        self.term_dir.read().is_some()
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
        
        // Full-load mode: merge into `data` immediately (before disk I/O) so that
        // searches succeed even while the background flush thread is writing to disk.
        // This eliminates the window where data is in neither buffer nor data.
        self.merge_into_data(&entries);
        
        // Build block data — serialize + compress each entry in parallel
        let num_entries = entries.len();
        let compressed_entries: Vec<Result<Vec<u8>, String>> = if num_entries > 64 {
            // Parallel path: each scoped thread processes a chunk and returns results
            let num_threads = std::thread::available_parallelism()
                .map(|p| p.get()).unwrap_or(4).min(num_entries).max(1);
            let chunk_size = (num_entries + num_threads - 1) / num_threads;
            let entries_ref = &entries;
            let chunk_results: Vec<Vec<(usize, Result<Vec<u8>, String>)>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..num_threads).filter_map(|t| {
                    let start = t * chunk_size;
                    if start >= num_entries { return None; }
                    let end = (start + chunk_size).min(num_entries);
                    Some(s.spawn(move || {
                        let mut results = Vec::with_capacity(end - start);
                        for i in start..end {
                            let (_, bitmap) = &entries_ref[i];
                            let res = encode_posting(bitmap).map_err(|e| e.to_string());
                            results.push((i, res));
                        }
                        results
                    }))
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            // Assemble results in original order
            let mut all_results: Vec<Result<Vec<u8>, String>> = (0..num_entries).map(|_| Ok(Vec::new())).collect();
            for chunk in chunk_results {
                for (idx, res) in chunk {
                    all_results[idx] = res;
                }
            }
            all_results
        } else {
            // Sequential path for small batches
            entries.iter().map(|(_, bitmap)| {
                encode_posting(bitmap).map_err(|e| e.to_string())
            }).collect()
        };

        let mut block = Vec::new();
        // term_count
        block.extend_from_slice(&(num_entries as u32).to_le_bytes());

        for (i, (term, _)) in entries.iter().enumerate() {
            // term_len + term
            block.extend_from_slice(&(term.len() as u16).to_le_bytes());
            block.extend_from_slice(term.as_bytes());

            let compressed = compressed_entries[i].as_ref()
                .map_err(|e| LsmSingleError::CompressionError(e.clone()))?;

            // data_len + data
            block.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            block.extend_from_slice(compressed);
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
        let crc = Self::compute_payload_crc(&mut file)?;
        header.set_payload_crc(crc);
        
        Self::write_header(&mut file, &header)?;
        file.sync_all()?; // Ensure data + metadata are on disk
        drop(header);
        drop(file);
        // Durably record the directory entry for newly-created index files as well.
        crate::persist::sync_parent_dir(&self.path)?;
        
        // Update in-memory index structures based on mode (full-load already done above)
        if self.lazy_load {
            // Lazy load mode: update index directory and clear cache
            // Rescan file to get new offsets
            let mut file = self.file.write();
            let header = self.header.read();
            if let Ok(new_index) = Self::load_term_index(&mut file, &header) {
                drop(header);
                drop(file);
                
                // Replace term_index with new complete index
                // (load_term_index already scans all blocks and collects all locations)
                *self.term_index.write() = new_index;
            }
            
            // Clear cache to ensure subsequent searches load fresh data from term_index
            // This is necessary because term_index now contains complete data from all blocks,
            // while cache may only have partial data
            self.cache.lock().clear();

            // The `.tdir` sidecar is a snapshot from the last compact. Appending a new
            // block makes it stale (e.g. updated docs re-added to existing Chinese terms
            // would be invisible if we kept serving `.tdir`). Drop it so `get_lazy`
            // falls back to the freshly rebuilt `term_index` until the next compact.
            *self.term_dir.write() = None;
            self.needs_rebuild.store(true, Ordering::SeqCst);
            let tdir = self.tdir_path();
            if tdir.exists() {
                let _ = std::fs::remove_file(&tdir);
            }
        }
        // Full load mode: `data` was already updated before disk I/O (see above)
        
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
    pub fn compact_with_deletions(&self, deleted_docs: &[u64]) -> Result<(), LsmSingleError> {
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
        
        // Snapshot of the fully-compacted live postings, used below to (re)build the
        // sorted mmap term directory sidecar for lazy-load mode. Only collected in
        // lazy_load mode since that's the only mode that consults the sidecar.
        let mut live_entries: Vec<(String, FastBitmap)> = Vec::new();
        
        {
            // Disable WAL for temp index as this is atomic operation, no recovery needed
            let new_index = Self::create_with_options(&tmp_path, false)?;
            
            // Get all terms based on mode
            if self.lazy_load {
                // Lazy load mode: need to get data from index directory and cache
                let term_index = self.term_index.read();
                live_entries.reserve(term_index.len());
                for (term, entry) in term_index.iter() {
                    // Load bitmap
                    if let Some(mut bitmap) = self.load_term_from_file(entry) {
                        // Apply deletions
                        for &doc_id in deleted_docs {
                            bitmap.remove(doc_id);
                        }
                        
                        // Only write non-empty bitmaps
                        if !bitmap.is_empty() {
                            new_index.buffer.write().insert(term.clone(), bitmap.clone());
                            live_entries.push((term.clone(), bitmap));
                        }
                    }
                }
            } else {
                // Full load mode: copy from in-memory DashMap
                for entry in self.data.iter() {
                    let mut new_bitmap = entry.value().clone();
                    
                    // Apply deletions
                    for &doc_id in deleted_docs {
                        new_bitmap.remove(doc_id);
                    }
                    
                    // Only write non-empty bitmaps
                    if !new_bitmap.is_empty() {
                        new_index.buffer.write().insert(entry.key().clone(), new_bitmap);
                    }
                }
            }
            new_index.flush()?;
            // `flush` already sync_all'd the temp index file. Drop the handle so
            // rename can replace it on all platforms (notably Windows).
        }
        
        // Atomic replace + parent-dir fsync so the new index inode is durable.
        crate::persist::durable_rename(&tmp_path, &self.path)?;
        
        // Clean up potentially remaining temp WAL file
        let tmp_wal_path = {
            let mut p = tmp_path.clone().into_os_string();
            p.push(".wal");
            PathBuf::from(p)
        };
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
            // Rebuild the sorted mmap term directory sidecar from the compacted live
            // postings so lazy-load lookups can use the fast binary-search path again.
            // Sidecar write itself is durable (tmp + sync + parent fsync).
            self.write_term_dir_sidecar(&live_entries);
        } else {
            let new_data = Self::load_all_blocks(&mut file, &new_header)?;
            self.data.clear();
            for (k, v) in new_data {
                self.data.insert(k, v);
            }
        }
        
        *self.file.write() = file;
        *self.header.write() = new_header;
        
        Ok(())
    }
    
    pub fn term_count(&self) -> usize {
        let persisted_count = if self.lazy_load {
            self.term_index.read().len()
        } else {
            self.data.len()
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
            terms.extend(self.data.iter().map(|r| r.key().clone()));
        }
        
        // Add buffer terms
        terms.extend(self.buffer.read().keys().cloned());
        terms.into_iter().collect()
    }
    
    /// Remove specified document IDs from all data
    pub fn remove_docs(&self, doc_ids: &[u64]) {
        // Remove from buffer
        for entry in self.buffer.write().values_mut() {
            for &doc_id in doc_ids {
                entry.remove(doc_id);
            }
        }
        
        // Remove from loaded data
        self.data.iter_mut().for_each(|mut entry| {
            for &doc_id in doc_ids {
                entry.value_mut().remove(doc_id);
            }
        });
    }
}

impl Drop for LsmSingleIndex {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_header_roundtrip_le_layout() {
        let header = FileHeader {
            magic: *MAGIC,
            version: VERSION,
            flags: 0xAABB,
            block_count: 0x1122334455667788,
            total_terms: 0x99AABBCCDDEEFF00,
            total_docs: 0x0102030405060708,
            _reserved: [0x5A; 32],
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(MAGIC, b"NFS2");
        assert_eq!(LEGACY_MAGIC, b"NFS1");
        assert_eq!(&bytes[4..6], &VERSION.to_le_bytes());
        assert_eq!(&bytes[6..8], &0xAABBu16.to_le_bytes());
        assert_eq!(&bytes[8..16], &0x1122334455667788u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &0x99AABBCCDDEEFF00u64.to_le_bytes());
        assert_eq!(&bytes[24..32], &0x0102030405060708u64.to_le_bytes());
        assert_eq!(&bytes[32..64], &[0x5Au8; 32]);

        let decoded = FileHeader::from_bytes(&bytes);
        assert_eq!(decoded.magic, header.magic);
        assert_eq!(decoded.version, header.version);
        assert_eq!(decoded.flags, header.flags);
        assert_eq!(decoded.block_count, header.block_count);
        assert_eq!(decoded.total_terms, header.total_terms);
        assert_eq!(decoded.total_docs, header.total_docs);
        assert_eq!(decoded._reserved, header._reserved);
    }

    #[test]
    fn test_analyzer_version_roundtrip() {
        let mut header = FileHeader {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            block_count: 0,
            total_terms: 0,
            total_docs: 0,
            _reserved: [0; 32],
        };
        header.set_analyzer_version(ANALYZER_VERSION);

        let bytes = header.to_bytes();
        assert_eq!(&bytes[32..36], &ANALYZER_VERSION.to_le_bytes());

        let decoded = FileHeader::from_bytes(&bytes);
        assert_eq!(decoded.analyzer_version(), ANALYZER_VERSION);
    }

    #[test]
    fn test_posting_single_doc_codec() {
        let bitmap = FastBitmap::from_iter([u32::MAX as u64 + 9]);
        let compressed = encode_posting(&bitmap).unwrap();
        let decoded = decode_posting(&compressed).unwrap();
        assert_eq!(decoded.to_vec(), vec![u32::MAX as u64 + 9]);
    }

    #[test]
    fn test_posting_vbyte_and_roaring_codecs() {
        let small = FastBitmap::from_iter([1u64, 2, 5, 10]);
        let decoded_small = decode_posting(&encode_posting(&small).unwrap()).unwrap();
        assert_eq!(decoded_small.len(), 4);

        let large_ids: Vec<u64> = (0..200).collect();
        let large = FastBitmap::from_iter(large_ids);
        let decoded_large = decode_posting(&encode_posting(&large).unwrap()).unwrap();
        assert_eq!(decoded_large.len(), 200);
    }

    #[test]
    fn test_payload_crc_roundtrip_on_flush() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crc.nfts");
        {
            let index = LsmSingleIndex::create_with_options(&path, false).unwrap();
            index.upsert("hello", 1);
            index.upsert("world", 2);
            index.flush().unwrap();
        }
        // Reopen should verify CRC successfully
        let index = LsmSingleIndex::open_with_options(&path, false).unwrap();
        assert!(index.get("hello").unwrap().contains(1));
    }

    #[test]
    fn test_compact_then_reopen_lazy_uses_term_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy_tdir.nfts");

        // Build up an index with a handful of terms/docs, then compact so the
        // sorted mmap term directory sidecar gets written.
        {
            let index = LsmSingleIndex::create_full_options(&path, false, true, 1000).unwrap();
            index.upsert("hello", 1);
            index.upsert("world", 2);
            index.upsert("hello", 3);
            index.upsert("rust", 4);
            index.flush().unwrap();
            index.compact().unwrap();

            // The sidecar should be built and loaded immediately after compact.
            assert!(index.term_dir_loaded(), "term directory should be loaded right after compact");
            assert!(!index.needs_rebuild());
        }

        // Reopen fresh (simulating a process restart) in lazy mode and verify the
        // sidecar is picked back up and lookups go through it correctly.
        let reopened = LsmSingleIndex::open_full_options(&path, false, true, 1000).unwrap();
        assert!(reopened.term_dir_loaded(), "term directory should load on reopen");
        assert!(!reopened.needs_rebuild());

        let hello = reopened.get("hello").unwrap();
        assert!(hello.contains(1));
        assert!(hello.contains(3));
        assert_eq!(hello.len(), 2);

        let world = reopened.get("world").unwrap();
        assert!(world.contains(2));

        let rust = reopened.get("rust").unwrap();
        assert!(rust.contains(4));

        assert!(reopened.get("missing-term").is_none());

        // Cache should have been populated via the directory fast path.
        let (_, misses, cache_len) = reopened.cache_stats();
        assert!(misses >= 3);
        assert!(cache_len >= 3);
    }

    #[test]
    fn test_open_lazy_without_sidecar_falls_back_gracefully() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy_no_tdir.nfts");

        {
            let index = LsmSingleIndex::create_full_options(&path, false, true, 1000).unwrap();
            index.upsert("alpha", 1);
            index.flush().unwrap();
            // No compact() call here, so no sidecar is ever written.
        }

        let reopened = LsmSingleIndex::open_full_options(&path, false, true, 1000).unwrap();
        assert!(!reopened.term_dir_loaded(), "no sidecar should be present without a compact");
        assert!(!reopened.needs_rebuild(), "missing sidecar is not an error condition");
        assert!(reopened.get("alpha").unwrap().contains(1));
    }
}
