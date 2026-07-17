//! Write-Ahead Log (WAL) Module — NWA2 (u64 doc IDs)
//!
//! ## File Format
//!
//! ```text
//! [Header 32B]
//!   magic: 4B "NWA2"
//!   version: 2B
//!   flags: 2B
//!   sequence: 8B
//!   batch_count: 8B
//!   checksum: 4B
//!   _reserved: 4B
//!
//! [Batch...]
//!   batch_len: 4B
//!   entry_count: 4B
//!   [Entry...]
//!     op: 1B (0=add, 1=remove)
//!     term_len: 2B
//!     term: [term_len]B
//!     doc_id: 8B (little-endian u64)
//!   batch_checksum: 4B (CRC32)
//! ```

use crc32fast::Hasher;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const WAL_MAGIC: &[u8; 4] = b"NWA2";
const WAL_LEGACY_MAGIC: &[u8; 4] = b"NWAL";
const WAL_VERSION: u16 = 2;
const WAL_HEADER_SIZE: usize = 32;
const MAX_WAL_BATCH_BYTES: usize = 64 * 1024 * 1024;

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
    pub doc_id: u64,
}

/// WAL batch
#[derive(Clone, Debug, Default)]
pub struct WalBatch {
    pub entries: Vec<WalEntry>,
}

impl WalBatch {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, term: String, doc_id: u64) {
        self.entries.push(WalEntry {
            op: WalOp::Add,
            term,
            doc_id,
        });
    }

    pub fn remove(&mut self, term: String, doc_id: u64) {
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

/// WAL file header (portable little-endian encoding)
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
        let bytes = self.to_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&bytes[..28]);
        self.checksum = hasher.finalize();
    }

    fn verify_checksum(&self) -> bool {
        let mut copy = *self;
        copy.checksum = 0;
        let bytes = copy.to_bytes();
        let mut hasher = Hasher::new();
        hasher.update(&bytes[..28]);
        hasher.finalize() == self.checksum
    }

    fn to_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.sequence.to_le_bytes());
        buf[16..24].copy_from_slice(&self.batch_count.to_le_bytes());
        buf[24..28].copy_from_slice(&self.checksum.to_le_bytes());
        buf[28..32].copy_from_slice(&self._reserved.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8; WAL_HEADER_SIZE]) -> Self {
        Self {
            magic: bytes[0..4].try_into().unwrap(),
            version: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            flags: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            sequence: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            batch_count: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            _reserved: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        }
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
        let mut path = index_path.as_ref().to_path_buf();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("index");
        path.set_file_name(format!("{}.wal", file_name));

        let (file, header) = if path.exists() {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)?;

            let header = Self::read_header(&mut file)?;
            if &header.magic == WAL_LEGACY_MAGIC {
                return Err(WalError::InvalidFormat(
                    "Legacy NWAL format is not supported; rebuild the index".into(),
                ));
            }
            if &header.magic != WAL_MAGIC {
                return Err(WalError::InvalidFormat("Invalid magic".into()));
            }
            if header.version != WAL_VERSION {
                return Err(WalError::InvalidFormat(format!(
                    "Unsupported WAL version {}",
                    header.version
                )));
            }
            if !header.verify_checksum() {
                return Err(WalError::ChecksumMismatch);
            }

            (file, header)
        } else {
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
        file.write_all(&header.to_bytes())?;
        Ok(())
    }

    pub fn log_add(&self, term: &str, doc_id: u64) {
        self.current_batch.lock().add(term.to_string(), doc_id);
    }

    pub fn log_remove(&self, term: &str, doc_id: u64) {
        self.current_batch.lock().remove(term.to_string(), doc_id);
    }

    pub fn log_add_batch(&self, entries: &[(String, u64)]) {
        let mut batch = self.current_batch.lock();
        for (term, doc_id) in entries {
            batch.add(term.clone(), *doc_id);
        }
    }

    pub fn commit(&self) -> Result<usize, WalError> {
        let mut batch = self.current_batch.lock();
        if batch.is_empty() {
            return Ok(0);
        }

        let entries_count = batch.len();
        let batch_data = self.serialize_batch(&batch)?;
        batch.clear();
        drop(batch);

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

        data.extend_from_slice(&[0u8; 4]);

        let entry_count = batch.entries.len() as u32;
        data.extend_from_slice(&entry_count.to_le_bytes());
        hasher.update(&entry_count.to_le_bytes());

        for entry in &batch.entries {
            data.push(entry.op as u8);
            hasher.update(&[entry.op as u8]);

            let term_bytes = entry.term.as_bytes();
            let term_len = term_bytes.len() as u16;
            data.extend_from_slice(&term_len.to_le_bytes());
            data.extend_from_slice(term_bytes);
            hasher.update(&term_len.to_le_bytes());
            hasher.update(term_bytes);

            data.extend_from_slice(&entry.doc_id.to_le_bytes());
            hasher.update(&entry.doc_id.to_le_bytes());
        }

        let checksum = hasher.finalize();
        data.extend_from_slice(&checksum.to_le_bytes());

        let batch_len = (data.len() - 4) as u32;
        data[0..4].copy_from_slice(&batch_len.to_le_bytes());

        Ok(data)
    }

    /// Recover batches; truncates dirty tail on corruption (ApexFTS-style repair).
    pub fn recover(&self) -> Result<Vec<WalBatch>, WalError> {
        let header = self.header.lock();
        if header.batch_count == 0 {
            return Ok(Vec::new());
        }
        let expected_batches = header.batch_count;
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
        let mut valid_len = WAL_HEADER_SIZE as u64;
        let mut truncated = false;

        for _ in 0..expected_batches {
            match self.read_batch(file, offset) {
                Ok((batch, next_offset)) => {
                    batches.push(batch);
                    offset = next_offset;
                    valid_len = next_offset;
                }
                Err(WalError::CorruptedBatch(_)) => {
                    truncated = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        if truncated || batches.len() as u64 != expected_batches {
            file.set_len(valid_len)?;
            file.sync_data()?;
            let mut header = self.header.lock();
            header.batch_count = batches.len() as u64;
            header.sequence += 1;
            header.update_checksum();
            Self::write_header(file, &header)?;
            file.sync_data()?;
        }

        Ok(batches)
    }

    fn read_batch(&self, file: &mut File, offset: u64) -> Result<(WalBatch, u64), WalError> {
        file.seek(SeekFrom::Start(offset))?;

        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            return Err(WalError::CorruptedBatch(offset));
        }
        let batch_len = u32::from_le_bytes(len_buf) as usize;
        if batch_len == 0 || batch_len > MAX_WAL_BATCH_BYTES {
            return Err(WalError::CorruptedBatch(offset));
        }

        let mut batch_data = vec![0u8; batch_len];
        if file.read_exact(&mut batch_data).is_err() {
            return Err(WalError::CorruptedBatch(offset));
        }

        if batch_data.len() < 8 {
            return Err(WalError::CorruptedBatch(offset));
        }

        let checksum_offset = batch_data.len() - 4;
        let stored_checksum =
            u32::from_le_bytes(batch_data[checksum_offset..].try_into().unwrap());

        let mut hasher = Hasher::new();
        hasher.update(&batch_data[..checksum_offset]);
        if hasher.finalize() != stored_checksum {
            return Err(WalError::CorruptedBatch(offset));
        }

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

            let op = WalOp::try_from(data[offset])?;
            offset += 1;

            if offset + 2 > data.len() {
                break;
            }
            let term_len = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;

            if offset + term_len > data.len() {
                break;
            }
            let term = String::from_utf8_lossy(&data[offset..offset + term_len]).to_string();
            offset += term_len;

            if offset + 8 > data.len() {
                break;
            }
            let doc_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;

            batch.entries.push(WalEntry { op, term, doc_id });
        }

        Ok(batch)
    }

    pub fn clear(&self) -> Result<(), WalError> {
        let mut file_guard = self.file.lock();
        let file = file_guard.as_mut().ok_or_else(|| {
            WalError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL file not open",
            ))
        })?;

        file.set_len(WAL_HEADER_SIZE as u64)?;

        let mut header = self.header.lock();
        header.batch_count = 0;
        header.sequence += 1;
        header.update_checksum();

        Self::write_header(file, &header)?;
        file.sync_all()?;

        self.current_batch.lock().clear();

        Ok(())
    }

    pub fn file_size(&self) -> Result<u64, WalError> {
        let file_guard = self.file.lock();
        if let Some(ref file) = *file_guard {
            Ok(file.metadata()?.len())
        } else {
            Ok(0)
        }
    }

    pub fn pending_batch_count(&self) -> u64 {
        self.header.lock().batch_count
    }

    pub fn current_batch_size(&self) -> usize {
        self.current_batch.lock().len()
    }

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
    fn test_wal_header_roundtrip_le_layout() {
        let mut header = WalHeader {
            magic: *WAL_MAGIC,
            version: WAL_VERSION,
            flags: 0x0102,
            sequence: 0x1122334455667788,
            batch_count: 0x99AABBCCDDEEFF00,
            checksum: 0,
            _reserved: 0,
        };
        header.update_checksum();

        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..4], WAL_MAGIC);
        let decoded = WalHeader::from_bytes(&bytes);
        assert!(decoded.verify_checksum());
        assert_eq!(decoded.sequence, header.sequence);
    }

    #[test]
    fn test_wal_basic() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.nfts");

        {
            let wal = WriteAheadLog::open(&index_path).unwrap();
            wal.log_add("hello", 1);
            wal.log_add("world", u32::MAX as u64 + 17);
            wal.commit().unwrap();

            wal.log_add("foo", 3);
            wal.log_remove("hello", 1);
            wal.commit().unwrap();
        }

        {
            let wal = WriteAheadLog::open(&index_path).unwrap();
            let batches = wal.recover().unwrap();

            assert_eq!(batches.len(), 2);
            assert_eq!(batches[0].entries[1].doc_id, u32::MAX as u64 + 17);
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
        assert!(wal.recover().unwrap().is_empty());
    }

    #[test]
    fn test_wal_tail_truncation_repair() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.nfts");

        let wal = WriteAheadLog::open(&index_path).unwrap();
        wal.log_add("ok", 1);
        wal.commit().unwrap();

        // Append dirty bytes past a valid batch
        {
            let mut file_guard = wal.file.lock();
            let file = file_guard.as_mut().unwrap();
            let mut header = wal.header.lock();
            header.batch_count = 2; // claim a second batch that doesn't exist cleanly
            header.update_checksum();
            WriteAheadLog::write_header(file, &header).unwrap();
            file.seek(SeekFrom::End(0)).unwrap();
            file.write_all(&[0xFFu8; 16]).unwrap();
            file.sync_all().unwrap();
        }

        let batches = wal.recover().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(wal.pending_batch_count(), 1);

        // Future appends still work
        wal.log_add("next", 2);
        wal.commit().unwrap();
        assert_eq!(wal.recover().unwrap().len(), 2);
    }
}
