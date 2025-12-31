//! Shard Management Module
//! 
//! Implements high-performance sharded index storage and access

use crate::bitmap::FastBitmap;
use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// File format magic number
const MAGIC_HEADER: &[u8; 4] = b"NFTS";
const MAGIC_FOOTER: &[u8; 4] = b"STFN";
const FORMAT_VERSION: u16 = 0x0002;

/// Shard error types
#[derive(Error, Debug)]
pub enum ShardError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Format error: {0}")]
    FormatError(String),
    #[error("Checksum error")]
    ChecksumError,
    #[error("Shard not found: {0}")]
    ShardNotFound(String),
    #[error("Deserialization error: {0}")]
    DeserializeError(String),
}

/// Shard file header
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct ShardHeader {
    magic: [u8; 4],
    version: u16,
    status: u8,
    _reserved: u8,
    term_count: u32,
    total_size: u64,
    checksum: u32,
}

/// Term entry
#[derive(Debug, Clone)]
struct TermEntry {
    offset: u64,
    size: u32,
    compressed: bool,
}

/// Memory-mapped shard
/// 
/// Uses mmap for direct file access, avoiding copy overhead
pub struct MappedShard {
    path: PathBuf,
    mmap: Option<Mmap>,
    terms: FxHashMap<String, TermEntry>,
    term_count: usize,
}

impl MappedShard {
    /// Open shard file (read-only mode)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ShardError> {
        let path = path.as_ref().to_path_buf();
        
        if !path.exists() {
            return Ok(Self {
                path,
                mmap: None,
                terms: FxHashMap::default(),
                term_count: 0,
            });
        }
        
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Parse file header and index
        let (terms, term_count) = Self::parse_index(&mmap)?;
        
        Ok(Self {
            path,
            mmap: Some(mmap),
            terms,
            term_count,
        })
    }
    
    /// Parse index
    fn parse_index(data: &[u8]) -> Result<(FxHashMap<String, TermEntry>, usize), ShardError> {
        if data.len() < 24 {
            return Err(ShardError::FormatError("File too small".into()));
        }
        
        // Check magic number
        if &data[0..4] != MAGIC_HEADER {
            return Err(ShardError::FormatError("Invalid file header".into()));
        }
        
        // Read header info
        let version = u16::from_be_bytes([data[4], data[5]]);
        let _status = data[6];
        let term_count = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
        
        let mut terms = FxHashMap::with_capacity_and_hasher(term_count, Default::default());
        let mut pos = 24; // Skip file header
        
        // Read all term entries
        while pos < data.len() - 8 {
            // Read segment header (11 bytes)
            if pos + 11 > data.len() {
                break;
            }
            
            let segment_type = data[pos];
            if segment_type != 0x01 {
                break; // Encountered non-term segment
            }
            
            let compressed = data[pos + 1] != 0;
            let term_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let data_len = u32::from_be_bytes([data[pos + 6], data[pos + 7], data[pos + 8], data[pos + 9]]) as usize;
            
            pos += 11;
            
            // Read term name
            if pos + term_len > data.len() {
                break;
            }
            let term = String::from_utf8_lossy(&data[pos..pos + term_len]).to_string();
            pos += term_len;
            
            // Record data position
            let entry = TermEntry {
                offset: pos as u64,
                size: data_len as u32,
                compressed,
            };
            terms.insert(term, entry);
            
            pos += data_len;
            pos += 4; // Skip checksum
        }
        
        Ok((terms, term_count))
    }
    
    /// Get term's bitmap
    pub fn get_bitmap(&self, term: &str) -> Option<FastBitmap> {
        let entry = self.terms.get(term)?;
        let mmap = self.mmap.as_ref()?;
        
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        
        if end > mmap.len() {
            return None;
        }
        
        let data = &mmap[start..end];
        
        // Decompress (if needed)
        let decompressed = if entry.compressed {
            match lz4_flex::decompress_size_prepended(data) {
                Ok(d) => d,
                Err(_) => {
                    // Try zlib decompression
                    match miniz_oxide::inflate::decompress_to_vec_zlib(data) {
                        Ok(d) => d,
                        Err(_) => return None,
                    }
                }
            }
        } else {
            data.to_vec()
        };
        
        FastBitmap::deserialize(&decompressed).ok()
    }
    
    /// Check if term exists
    #[inline]
    pub fn contains_term(&self, term: &str) -> bool {
        self.terms.contains_key(term)
    }
    
    /// Get all terms
    pub fn terms(&self) -> impl Iterator<Item = &String> {
        self.terms.keys()
    }
    
    /// Get term count
    #[inline]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
    
    /// Get path
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Shard writer
pub struct ShardWriter {
    path: PathBuf,
    terms: FxHashMap<String, Vec<u8>>,
    compression_level: i32,
}

impl ShardWriter {
    /// Create shard writer
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            terms: FxHashMap::default(),
            compression_level: 1, // Fast compression
        }
    }
    
    /// Add term
    pub fn add_term(&mut self, term: String, bitmap: &FastBitmap) -> Result<(), ShardError> {
        let data = bitmap.serialize()
            .map_err(|e| ShardError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        self.terms.insert(term, data);
        Ok(())
    }
    
    /// Merge term
    pub fn merge_term(&mut self, term: String, bitmap: &FastBitmap) -> Result<(), ShardError> {
        let new_data = bitmap.serialize()
            .map_err(|e| ShardError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        
        if let Some(existing) = self.terms.get_mut(&term) {
            // Deserialize existing data
            if let Ok(mut existing_bitmap) = FastBitmap::deserialize(existing) {
                existing_bitmap.or_inplace(bitmap);
                *existing = existing_bitmap.serialize()
                    .map_err(|e| ShardError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
            }
        } else {
            self.terms.insert(term, new_data);
        }
        
        Ok(())
    }
    
    /// Write to file
    pub fn write(&self) -> Result<(), ShardError> {
        if self.terms.is_empty() {
            return Ok(());
        }
        
        // Create parent directory
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Use temp file for atomic write
        let temp_path = self.path.with_extension("tmp");
        
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
            
            // Write file header (placeholder first)
            let header_placeholder = [0u8; 24];
            writer.write_all(&header_placeholder)?;
            
            let mut total_size = 24u64;
            
            // Write terms sorted
            let mut sorted_terms: Vec<_> = self.terms.iter().collect();
            sorted_terms.sort_by_key(|(k, _)| *k);
            
            for (term, data) in &sorted_terms {
                // Compress large data
                let (compressed_data, is_compressed) = if data.len() > 1024 {
                    let compressed = lz4_flex::compress_prepend_size(data);
                    if compressed.len() < data.len() * 9 / 10 {
                        (compressed, true)
                    } else {
                        ((*data).clone(), false)
                    }
                } else {
                    ((*data).clone(), false)
                };
                
                let term_bytes = term.as_bytes();
                
                // Write segment header
                let segment_header = [
                    0x01u8,                                          // Segment type
                    if is_compressed { 1 } else { 0 },              // Compression flag
                    (term_bytes.len() >> 8) as u8,                   // Term length high byte
                    term_bytes.len() as u8,                          // Term length low byte
                    0, 0,                                            // Reserved
                    (compressed_data.len() >> 24) as u8,             // Data length
                    (compressed_data.len() >> 16) as u8,
                    (compressed_data.len() >> 8) as u8,
                    compressed_data.len() as u8,
                    0,                                               // Reserved
                ];
                writer.write_all(&segment_header)?;
                writer.write_all(term_bytes)?;
                writer.write_all(&compressed_data)?;
                
                // Calculate segment checksum
                let mut checksum_data = Vec::with_capacity(11 + term_bytes.len() + compressed_data.len());
                checksum_data.extend_from_slice(&segment_header);
                checksum_data.extend_from_slice(term_bytes);
                checksum_data.extend_from_slice(&compressed_data);
                let checksum = crc32fast::hash(&checksum_data);
                writer.write_all(&checksum.to_be_bytes())?;
                
                total_size += 11 + term_bytes.len() as u64 + compressed_data.len() as u64 + 4;
            }
            
            // Write file footer
            writer.write_all(MAGIC_FOOTER)?;
            writer.write_all(&0x12345678u32.to_be_bytes())?;
            total_size += 8;
            
            writer.flush()?;
            
            // Rewrite file header
            let file = writer.into_inner()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let mut file = file;
            file.seek(SeekFrom::Start(0))?;
            
            let header_data = [
                MAGIC_HEADER[0], MAGIC_HEADER[1], MAGIC_HEADER[2], MAGIC_HEADER[3],
                (FORMAT_VERSION >> 8) as u8, FORMAT_VERSION as u8,
                0x01, // Complete status
                0,    // Reserved
                (self.terms.len() >> 24) as u8,
                (self.terms.len() >> 16) as u8,
                (self.terms.len() >> 8) as u8,
                self.terms.len() as u8,
                (total_size >> 56) as u8,
                (total_size >> 48) as u8,
                (total_size >> 40) as u8,
                (total_size >> 32) as u8,
                (total_size >> 24) as u8,
                (total_size >> 16) as u8,
                (total_size >> 8) as u8,
                total_size as u8,
            ];
            let checksum = crc32fast::hash(&header_data);
            
            file.write_all(&header_data)?;
            file.write_all(&checksum.to_be_bytes())?;
            file.sync_all()?;
        }
        
        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;
        
        Ok(())
    }
}

/// Shard manager
/// 
/// Manages loading and access of multiple shards
pub struct ShardManager {
    shards: RwLock<FxHashMap<u32, Arc<MappedShard>>>,
    shard_dir: PathBuf,
    shard_bits: u32,
}

impl ShardManager {
    /// Create shard manager
    pub fn new<P: AsRef<Path>>(shard_dir: P, shard_bits: u32) -> Self {
        Self {
            shards: RwLock::new(FxHashMap::default()),
            shard_dir: shard_dir.as_ref().to_path_buf(),
            shard_bits,
        }
    }
    
    /// Calculate term's shard ID
    #[inline]
    pub fn get_shard_id(&self, term: &str) -> u32 {
        let hash = xxhash_rust::xxh32::xxh32(term.as_bytes(), 0);
        hash & ((1 << self.shard_bits) - 1)
    }
    
    /// Get shard
    pub fn get_shard(&self, shard_id: u32) -> Option<Arc<MappedShard>> {
        // Try to get from cache first
        {
            let shards = self.shards.read();
            if let Some(shard) = shards.get(&shard_id) {
                return Some(Arc::clone(shard));
            }
        }
        
        // Load shard
        let shard_path = self.shard_dir.join(format!("shard_{}.nfts", shard_id));
        if !shard_path.exists() {
            return None;
        }
        
        match MappedShard::open(&shard_path) {
            Ok(shard) => {
                let shard = Arc::new(shard);
                let mut shards = self.shards.write();
                shards.insert(shard_id, Arc::clone(&shard));
                Some(shard)
            }
            Err(_) => None,
        }
    }
    
    /// Get term's bitmap
    pub fn get_bitmap(&self, term: &str) -> Option<FastBitmap> {
        let shard_id = self.get_shard_id(term);
        let shard = self.get_shard(shard_id)?;
        shard.get_bitmap(term)
    }
    
    /// Reload all shards
    pub fn reload(&self) -> Result<(), ShardError> {
        let mut shards = self.shards.write();
        shards.clear();
        Ok(())
    }
    
    /// Get shard directory
    #[inline]
    pub fn shard_dir(&self) -> &Path {
        &self.shard_dir
    }
}

// Add miniz_oxide dependency for zlib decompression
mod miniz_oxide {
    pub mod inflate {
        pub fn decompress_to_vec_zlib(data: &[u8]) -> Result<Vec<u8>, ()> {
            // Simple zlib decompression implementation
            use std::io::Read;
            let mut decoder = flate2::read::ZlibDecoder::new(data);
            let mut result = Vec::new();
            decoder.read_to_end(&mut result).map_err(|_| ())?;
            Ok(result)
        }
    }
}

// Dependencies needed
use flate2;
use crc32fast;
