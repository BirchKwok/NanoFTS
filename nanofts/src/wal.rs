//! Write-Ahead Log (WAL) Module
//!
//! Similar to SQLite's journal file, provides crash recovery capability
//!
//! ## File Format
//!
//! ```text
//! [Header 32B]
//!   magic: 4B "NWAL"
//!   version: 2B
//!   flags: 2B
//!   sequence: 8B (incrementing sequence number)
//!   batch_count: 8B
//!   checksum: 4B (header checksum)
//!   _reserved: 4B
//!
//! [Batch...]
//!   batch_len: 4B (excluding this field)
//!   entry_count: 4B
//!   [Entry...]
//!     op: 1B (0=add, 1=remove)
//!     term_len: 2B
//!     term: [term_len]B
//!     doc_id: 4B
//!   batch_checksum: 4B (CRC32)
//! ```
//!
//! ## Recovery Process
//!
//! 1. Check if WAL file exists on open
//! 2. Verify header checksum
//! 3. Read batches one by one, verify checksums
//! 4. Replay valid batch operations to index
//! 5. Clear WAL

use crc32fast::Hasher;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const WAL_MAGIC: &[u8; 4] = b"NWAL";
const WAL_VERSION: u16 = 1;
const WAL_HEADER_SIZE: usize = 32;

#[derive(Error, Debug)]
pub enum WalError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid WAL format: {0}")]
    InvalidFormat(String),
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    #[error("Corrupted batch at offset {0}")]
    CorruptedBatch(u64),
}

/// WAL operation type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOp {
    Add = 0,
    Remove = 1,
}

impl TryFrom<u8> for WalOp {
    type Error = WalError;
    
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(WalOp::Add),
            1 => Ok(WalOp::Remove),
            _ => Err(WalError::InvalidFormat(format!("Unknown op: {}", value))),
        }
    }
}

/// WAL entry
#[derive(Clone, Debug)]
pub struct WalEntry {
    pub op: WalOp,
    pub term: String,
    pub doc_id: u32,
}

/// WAL batch
#[derive(Clone, Debug, Default)]
pub struct WalBatch {
    pub entries: Vec<WalEntry>,
}

impl WalBatch {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
    
    pub fn add(&mut self, term: String, doc_id: u32) {
        self.entries.push(WalEntry {
            op: WalOp::Add,
            term,
            doc_id,
        });
    }
    
    pub fn remove(&mut self, term: String, doc_id: u32) {
        self.entries.push(WalEntry {
            op: WalOp::Remove,
            term,
            doc_id,
        });
    }
    
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// WAL file header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct WalHeader {
    magic: [u8; 4],
    version: u16,
    flags: u16,
    sequence: u64,
    batch_count: u64,
    checksum: u32,
    _reserved: u32,
}

impl WalHeader {
    fn new() -> Self {
        let mut header = Self {
            magic: *WAL_MAGIC,
            version: WAL_VERSION,
            flags: 0,
            sequence: 0,
            batch_count: 0,
            checksum: 0,
            _reserved: 0,
        };
        header.update_checksum();
        header
    }
    
    fn update_checksum(&mut self) {
        self.checksum = 0;
        let bytes = self.as_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&bytes[..28]); // Exclude checksum and reserved
        self.checksum = hasher.finalize();
    }
    
    fn verify_checksum(&self) -> bool {
        let mut copy = *self;
        copy.checksum = 0;
        let bytes = copy.as_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&bytes[..28]);
        hasher.finalize() == self.checksum
    }
    
    fn as_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        unsafe { std::mem::transmute(*self) }
    }
    
    fn from_bytes(bytes: &[u8; WAL_HEADER_SIZE]) -> Self {
        unsafe { std::ptr::read(bytes.as_ptr() as *const Self) }
    }
}

/// Write-Ahead Log
pub struct WriteAheadLog {
    path: PathBuf,
    file: Mutex<Option<File>>,
    header: Mutex<WalHeader>,
    current_batch: Mutex<WalBatch>,
    sync_on_write: bool,
}

impl WriteAheadLog {
    /// Create or open WAL file
    pub fn open<P: AsRef<Path>>(index_path: P) -> Result<Self, WalError> {
        // Append .wal instead of replacing extension, avoid index.nfts.tmp becoming index.nfts.nfts.wal
        let mut path = index_path.as_ref().to_path_buf();
        let file_name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("index");
        path.set_file_name(format!("{}.wal", file_name));
        
        let (file, header) = if path.exists() {
            // Open existing WAL
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)?;
            
            let header = Self::read_header(&mut file)?;
            if &header.magic != WAL_MAGIC {
                return Err(WalError::InvalidFormat("Invalid magic".into()));
            }
            if !header.verify_checksum() {
                return Err(WalError::ChecksumMismatch);
            }
            
            (file, header)
        } else {
            // Create new WAL
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            
            let header = WalHeader::new();
            Self::write_header(&mut file, &header)?;
            file.flush()?;
            
            (file, header)
        };
        
        Ok(Self {
            path,
            file: Mutex::new(Some(file)),
            header: Mutex::new(header),
            current_batch: Mutex::new(WalBatch::new()),
            sync_on_write: true,
        })
    }
    
    /// Set whether to sync to disk on every write
    pub fn set_sync_on_write(&mut self, sync: bool) {
        self.sync_on_write = sync;
    }
    
    fn read_header(file: &mut File) -> Result<WalHeader, WalError> {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut bytes)?;
        Ok(WalHeader::from_bytes(&bytes))
    }
    
    fn write_header(file: &mut File, header: &WalHeader) -> Result<(), WalError> {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.as_bytes())?;
        Ok(())
    }
    
    /// Add an add operation to current batch
    pub fn log_add(&self, term: &str, doc_id: u32) {
        self.current_batch.lock().add(term.to_string(), doc_id);
    }
    
    /// Add a remove operation to current batch
    pub fn log_remove(&self, term: &str, doc_id: u32) {
        self.current_batch.lock().remove(term.to_string(), doc_id);
    }
    
    /// Batch add operations
    pub fn log_add_batch(&self, entries: &[(String, u32)]) {
        let mut batch = self.current_batch.lock();
        for (term, doc_id) in entries {
            batch.add(term.clone(), *doc_id);
        }
    }
    
    /// Commit current batch to WAL file
    pub fn commit(&self) -> Result<usize, WalError> {
        let mut batch = self.current_batch.lock();
        if batch.is_empty() {
            return Ok(0);
        }
        
        let entries_count = batch.len();
        
        // Serialize batch
        let batch_data = self.serialize_batch(&batch)?;
        batch.clear();
        drop(batch);
        
        // Write to file
        let mut file_guard = self.file.lock();
        let file = file_guard.as_mut().ok_or_else(|| {
            WalError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL file not open",
            ))
        })?;
        
        file.seek(SeekFrom::End(0))?;
        file.write_all(&batch_data)?;
        
        if self.sync_on_write {
            file.sync_data()?;
        }
        
        // Update header
        let mut header = self.header.lock();
        header.batch_count += 1;
        header.sequence += 1;
        header.update_checksum();
        
        Self::write_header(file, &header)?;
        if self.sync_on_write {
            file.sync_data()?;
        }
        
        Ok(entries_count)
    }
    
    fn serialize_batch(&self, batch: &WalBatch) -> Result<Vec<u8>, WalError> {
        let mut data = Vec::new();
        let mut hasher = Hasher::new();
        
        // Reserve batch_len position
        data.extend_from_slice(&[0u8; 4]);
        
        // entry_count
        let entry_count = batch.entries.len() as u32;
        data.extend_from_slice(&entry_count.to_le_bytes());
        hasher.update(&entry_count.to_le_bytes());
        
        // entries
        for entry in &batch.entries {
            // op
            data.push(entry.op as u8);
            hasher.update(&[entry.op as u8]);
            
            // term_len + term
            let term_bytes = entry.term.as_bytes();
            let term_len = term_bytes.len() as u16;
            data.extend_from_slice(&term_len.to_le_bytes());
            data.extend_from_slice(term_bytes);
            hasher.update(&term_len.to_le_bytes());
            hasher.update(term_bytes);
            
            // doc_id
            data.extend_from_slice(&entry.doc_id.to_le_bytes());
            hasher.update(&entry.doc_id.to_le_bytes());
        }
        
        // checksum
        let checksum = hasher.finalize();
        data.extend_from_slice(&checksum.to_le_bytes());
        
        // Update batch_len (excluding the 4-byte field itself)
        let batch_len = (data.len() - 4) as u32;
        data[0..4].copy_from_slice(&batch_len.to_le_bytes());
        
        Ok(data)
    }
    
    /// Read all batches to recover
    pub fn recover(&self) -> Result<Vec<WalBatch>, WalError> {
        let header = self.header.lock();
        if header.batch_count == 0 {
            return Ok(Vec::new());
        }
        let batch_count = header.batch_count;
        drop(header);
        
        let mut file_guard = self.file.lock();
        let file = file_guard.as_mut().ok_or_else(|| {
            WalError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL file not open",
            ))
        })?;
        
        file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;
        
        let mut batches = Vec::new();
        let mut offset = WAL_HEADER_SIZE as u64;
        
        for _ in 0..batch_count {
            match self.read_batch(file, offset) {
                Ok((batch, next_offset)) => {
                    batches.push(batch);
                    offset = next_offset;
                }
                Err(WalError::CorruptedBatch(off)) => {
                    // Encountered corrupted batch, stop recovery
                    eprintln!("WAL: Corrupted batch at offset {}, stopping recovery", off);
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        
        Ok(batches)
    }
    
    fn read_batch(&self, file: &mut File, offset: u64) -> Result<(WalBatch, u64), WalError> {
        file.seek(SeekFrom::Start(offset))?;
        
        // batch_len
        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            return Err(WalError::CorruptedBatch(offset));
        }
        let batch_len = u32::from_le_bytes(len_buf) as usize;
        
        // Read batch data
        let mut batch_data = vec![0u8; batch_len];
        if file.read_exact(&mut batch_data).is_err() {
            return Err(WalError::CorruptedBatch(offset));
        }
        
        // Verify checksum
        if batch_data.len() < 8 {
            return Err(WalError::CorruptedBatch(offset));
        }
        
        let checksum_offset = batch_data.len() - 4;
        let stored_checksum = u32::from_le_bytes(
            batch_data[checksum_offset..].try_into().unwrap()
        );
        
        let mut hasher = Hasher::new();
        hasher.update(&batch_data[..checksum_offset]);
        let computed_checksum = hasher.finalize();
        
        if stored_checksum != computed_checksum {
            return Err(WalError::CorruptedBatch(offset));
        }
        
        // Parse batch
        let batch = self.parse_batch(&batch_data[..checksum_offset])?;
        
        let next_offset = offset + 4 + batch_len as u64;
        Ok((batch, next_offset))
    }
    
    fn parse_batch(&self, data: &[u8]) -> Result<WalBatch, WalError> {
        if data.len() < 4 {
            return Err(WalError::InvalidFormat("Batch too short".into()));
        }
        
        let entry_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut batch = WalBatch::new();
        
        for _ in 0..entry_count {
            if offset >= data.len() {
                break;
            }
            
            // op
            let op = WalOp::try_from(data[offset])?;
            offset += 1;
            
            // term_len
            if offset + 2 > data.len() {
                break;
            }
            let term_len = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap()) as usize;
            offset += 2;
            
            // term
            if offset + term_len > data.len() {
                break;
            }
            let term = String::from_utf8_lossy(&data[offset..offset+term_len]).to_string();
            offset += term_len;
            
            // doc_id
            if offset + 4 > data.len() {
                break;
            }
            let doc_id = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap());
            offset += 4;
            
            batch.entries.push(WalEntry { op, term, doc_id });
        }
        
        Ok(batch)
    }
    
    /// Clear WAL (call after successful flush to main index)
    pub fn clear(&self) -> Result<(), WalError> {
        let mut file_guard = self.file.lock();
        let file = file_guard.as_mut().ok_or_else(|| {
            WalError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL file not open",
            ))
        })?;
        
        // Truncate file
        file.set_len(WAL_HEADER_SIZE as u64)?;
        
        // Reset header
        let mut header = self.header.lock();
        header.batch_count = 0;
        header.sequence += 1;
        header.update_checksum();
        
        Self::write_header(file, &header)?;
        file.sync_all()?;
        
        // Clear current batch
        self.current_batch.lock().clear();
        
        Ok(())
    }
    
    /// Get WAL file size
    pub fn file_size(&self) -> Result<u64, WalError> {
        let file_guard = self.file.lock();
        if let Some(ref file) = *file_guard {
            Ok(file.metadata()?.len())
        } else {
            Ok(0)
        }
    }
    
    /// Get pending batch count
    pub fn pending_batch_count(&self) -> u64 {
        self.header.lock().batch_count
    }
    
    /// Get current batch entry count
    pub fn current_batch_size(&self) -> usize {
        self.current_batch.lock().len()
    }
    
    /// Delete WAL file
    pub fn remove(self) -> Result<(), WalError> {
        drop(self.file.lock().take());
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        // Try to commit uncommitted batch
        if !self.current_batch.lock().is_empty() {
            let _ = self.commit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_wal_basic() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.nfts");
        
        // Create WAL and write data
        {
            let wal = WriteAheadLog::open(&index_path).unwrap();
            wal.log_add("hello", 1);
            wal.log_add("world", 2);
            wal.commit().unwrap();
            
            wal.log_add("foo", 3);
            wal.log_remove("hello", 1);
            wal.commit().unwrap();
        }
        
        // Reopen and recover
        {
            let wal = WriteAheadLog::open(&index_path).unwrap();
            let batches = wal.recover().unwrap();
            
            assert_eq!(batches.len(), 2);
            assert_eq!(batches[0].entries.len(), 2);
            assert_eq!(batches[1].entries.len(), 2);
            
            assert_eq!(batches[0].entries[0].term, "hello");
            assert_eq!(batches[0].entries[0].op, WalOp::Add);
            
            assert_eq!(batches[1].entries[1].term, "hello");
            assert_eq!(batches[1].entries[1].op, WalOp::Remove);
        }
    }
    
    #[test]
    fn test_wal_clear() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.nfts");
        
        let wal = WriteAheadLog::open(&index_path).unwrap();
        wal.log_add("test", 1);
        wal.commit().unwrap();
        
        assert_eq!(wal.pending_batch_count(), 1);
        
        wal.clear().unwrap();
        
        assert_eq!(wal.pending_batch_count(), 0);
        let batches = wal.recover().unwrap();
        assert!(batches.is_empty());
    }
}
