//! Sorted, mmap-friendly term directory sidecar (`.tdir`) for the LSM lazy-load
//! fast path.
//!
//! The sidecar is a self-contained file (no cross-references into the main
//! `.nfts` file) built from the fully-compacted set of live postings. Its
//! directory is sorted by term bytes so a lookup is a binary search directly
//! over the memory-mapped file — no per-term `String` allocation or full
//! directory scan is required to answer `get(term)`.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! [magic: 4B "NTDR"][version: u32][term_count: u64]      <- 16B header
//! [directory: term_count * 24B]                          <- sorted by term bytes
//!     entry: term_off:u64, term_len:u32, _pad:u32, meta_off:u64
//! [term_data: raw UTF-8 term bytes]                       <- referenced by (term_off, term_len)
//! [postings_data: for each term, data_len:u32 + compressed posting bytes]
//!     (compressed bytes are the same format produced by `encode_posting`)
//! ```
//!
//! `term_off` and `meta_off` are absolute byte offsets into the sidecar file,
//! so a lookup only needs the mapped bytes and the parsed directory (which is
//! tiny: 24 bytes/term) — term bytes and posting bytes themselves are read
//! straight out of the OS page cache via the memory map.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub const TDIR_MAGIC: &[u8; 4] = b"NTDR";
pub const TDIR_VERSION: u32 = 1;
const HEADER_SIZE: usize = 16;
const ENTRY_SIZE: usize = 24;

#[derive(Clone, Copy, Debug)]
struct DirEntry {
    term_off: u64,
    term_len: u32,
    meta_off: u64,
}

/// Write a sorted term directory sidecar to `path`.
///
/// `sorted_entries` must already be sorted by term bytes (ascending) and
/// contain each term's already-compressed posting bytes (as produced by
/// `lsm_single::encode_posting`).
pub fn write_term_dir<P: AsRef<Path>>(
    path: P,
    sorted_entries: &[(String, Vec<u8>)],
) -> io::Result<()> {
    debug_assert!(
        sorted_entries
            .windows(2)
            .all(|w| w[0].0.as_bytes() <= w[1].0.as_bytes()),
        "sorted_entries must be sorted by term bytes"
    );

    let term_count = sorted_entries.len();
    let dir_len = term_count * ENTRY_SIZE;
    let term_data_start = HEADER_SIZE + dir_len;

    let mut term_data = Vec::new();
    let mut term_offsets = Vec::with_capacity(term_count);
    for (term, _) in sorted_entries {
        term_offsets.push(term_data_start as u64 + term_data.len() as u64);
        term_data.extend_from_slice(term.as_bytes());
    }

    let postings_data_start = term_data_start + term_data.len();
    let mut postings_data = Vec::new();
    let mut meta_offsets = Vec::with_capacity(term_count);
    for (_, compressed) in sorted_entries {
        meta_offsets.push(postings_data_start as u64 + postings_data.len() as u64);
        postings_data.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        postings_data.extend_from_slice(compressed);
    }

    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(TDIR_MAGIC);
    header.extend_from_slice(&TDIR_VERSION.to_le_bytes());
    header.extend_from_slice(&(term_count as u64).to_le_bytes());

    let mut dir_buf = Vec::with_capacity(dir_len);
    for (i, (term, _)) in sorted_entries.iter().enumerate() {
        dir_buf.extend_from_slice(&term_offsets[i].to_le_bytes());
        dir_buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
        dir_buf.extend_from_slice(&0u32.to_le_bytes()); // padding, keeps meta_off 8B-aligned
        dir_buf.extend_from_slice(&meta_offsets[i].to_le_bytes());
    }

    // Write to a temp file first, fsync, then durable-rename so a reader never
    // observes a partially-written sidecar and the directory entry itself survives crash.
    let path = path.as_ref();
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = std::path::PathBuf::from(tmp);
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(&header)?;
        file.write_all(&dir_buf)?;
        file.write_all(&term_data)?;
        file.write_all(&postings_data)?;
        file.sync_all()?;
    }
    crate::persist::durable_rename(&tmp_path, path)?;
    Ok(())
}

/// A memory-mapped, sorted term directory that supports binary-search lookups
/// without loading term strings or posting bytes into the heap upfront.
pub struct TermDirectory {
    mmap: Mmap,
    entries: Vec<DirEntry>,
}

impl TermDirectory {
    /// Open and validate a `.tdir` sidecar, memory-mapping its contents and
    /// eagerly parsing only the (small, fixed-width) directory entries.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tdir file too small"));
        }
        if &mmap[0..4] != TDIR_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid tdir magic"));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != TDIR_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported tdir version {version}"),
            ));
        }
        let term_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let dir_len = term_count * ENTRY_SIZE;
        if mmap.len() < HEADER_SIZE + dir_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated tdir directory"));
        }

        let mut entries = Vec::with_capacity(term_count);
        let mut offset = HEADER_SIZE;
        for _ in 0..term_count {
            let term_off = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
            let term_len = u32::from_le_bytes(mmap[offset + 8..offset + 12].try_into().unwrap());
            let meta_off = u64::from_le_bytes(mmap[offset + 16..offset + 24].try_into().unwrap());
            let term_end = term_off as usize + term_len as usize;
            if term_end > mmap.len() || (meta_off as usize) + 4 > mmap.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "tdir entry out of bounds"));
            }
            entries.push(DirEntry { term_off, term_len, meta_off });
            offset += ENTRY_SIZE;
        }

        Ok(Self { mmap, entries })
    }

    #[inline]
    fn term_bytes(&self, entry: &DirEntry) -> &[u8] {
        let start = entry.term_off as usize;
        let end = start + entry.term_len as usize;
        &self.mmap[start..end]
    }

    /// Binary search for `term`; returns the (still zstd-compressed) posting
    /// bytes if found, ready to pass to `lsm_single::decode_posting`.
    pub fn get(&self, term: &str) -> Option<&[u8]> {
        let target = term.as_bytes();
        let idx = self
            .entries
            .binary_search_by(|entry| self.term_bytes(entry).cmp(target))
            .ok()?;
        let entry = &self.entries[idx];
        let meta_start = entry.meta_off as usize;
        let data_len =
            u32::from_le_bytes(self.mmap[meta_start..meta_start + 4].try_into().unwrap()) as usize;
        let data_start = meta_start + 4;
        let data_end = data_start + data_len;
        if data_end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[data_start..data_end])
    }

    /// Number of terms in the directory.
    pub fn term_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(terms: &[&str]) -> Vec<(String, Vec<u8>)> {
        let mut sorted: Vec<&str> = terms.to_vec();
        sorted.sort_unstable();
        sorted
            .into_iter()
            .map(|t| (t.to_string(), format!("payload:{t}").into_bytes()))
            .collect()
    }

    #[test]
    fn write_and_lookup_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("terms.tdir");
        let data = entries(&["zebra", "apple", "mango", "a", "app"]);

        write_term_dir(&path, &data).unwrap();
        let term_dir = TermDirectory::open(&path).unwrap();

        assert_eq!(term_dir.term_count(), data.len());
        for (term, payload) in &data {
            let found = term_dir.get(term).expect("term should be present");
            assert_eq!(found, payload.as_slice());
        }
        assert!(term_dir.get("missing-term").is_none());
    }

    #[test]
    fn empty_directory_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.tdir");
        write_term_dir(&path, &[]).unwrap();
        let term_dir = TermDirectory::open(&path).unwrap();
        assert_eq!(term_dir.term_count(), 0);
        assert!(term_dir.get("anything").is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.tdir");
        std::fs::write(&path, b"NOPE0000000000000000000000000000").unwrap();
        assert!(TermDirectory::open(&path).is_err());
    }
}
