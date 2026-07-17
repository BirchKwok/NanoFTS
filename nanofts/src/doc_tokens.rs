//! Packed per-document token streams (CSR) for BM25 and phrase queries.

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::io::{Read, Write};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DocTokensError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Corrupt tokens blob: {0}")]
    Corrupt(String),
}

/// CSR-packed document tokens: `doc_ids` sorted, `offsets[i]..offsets[i+1]` is doc i's tokens.
#[derive(Clone, Default, Debug)]
pub struct PackedDocTokens {
    pub doc_ids: Vec<u64>,
    pub offsets: Vec<u64>,
    pub tokens: Vec<u32>,
}

impl PackedDocTokens {
    pub fn get(&self, doc_id: u64) -> Option<&[u32]> {
        let index = self.doc_ids.binary_search(&doc_id).ok()?;
        Some(&self.tokens[self.offsets[index] as usize..self.offsets[index + 1] as usize])
    }

    pub fn from_records(mut records: Vec<(u64, Box<[u32]>)>) -> Self {
        records.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        let token_count = records.iter().map(|(_, tokens)| tokens.len()).sum();
        let mut packed = Self {
            doc_ids: Vec::with_capacity(records.len()),
            offsets: Vec::with_capacity(records.len() + 1),
            tokens: Vec::with_capacity(token_count),
        };
        packed.offsets.push(0);
        for (doc_id, tokens) in records {
            packed.doc_ids.push(doc_id);
            packed.tokens.extend_from_slice(&tokens);
            packed.offsets.push(packed.tokens.len() as u64);
        }
        packed
    }

    pub fn total_terms(&self) -> u64 {
        self.tokens.iter().filter(|&&token| token != 0).count() as u64
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_ids.len() as u64
    }

    /// Serialize to a portable little-endian blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.doc_ids.len() as u64).to_le_bytes());
        for id in &self.doc_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&(self.offsets.len() as u64).to_le_bytes());
        for off in &self.offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        out.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        for tok in &self.tokens {
            out.extend_from_slice(&tok.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(mut data: &[u8]) -> Result<Self, DocTokensError> {
        let read_u64 = |data: &mut &[u8]| -> Result<u64, DocTokensError> {
            if data.len() < 8 {
                return Err(DocTokensError::Corrupt("truncated".into()));
            }
            let mut buf = [0u8; 8];
            data.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        };
        let n_docs = read_u64(&mut data)? as usize;
        let mut doc_ids = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            doc_ids.push(read_u64(&mut data)?);
        }
        let n_off = read_u64(&mut data)? as usize;
        let mut offsets = Vec::with_capacity(n_off);
        for _ in 0..n_off {
            offsets.push(read_u64(&mut data)?);
        }
        let n_tok = read_u64(&mut data)? as usize;
        if data.len() < n_tok * 4 {
            return Err(DocTokensError::Corrupt("token truncated".into()));
        }
        let mut tokens = Vec::with_capacity(n_tok);
        for _ in 0..n_tok {
            let mut buf = [0u8; 4];
            data.read_exact(&mut buf)?;
            tokens.push(u32::from_le_bytes(buf));
        }
        Ok(Self {
            doc_ids,
            offsets,
            tokens,
        })
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), DocTokensError> {
        w.write_all(&self.to_bytes())?;
        Ok(())
    }
}

/// Live token view: immutable base pack + per-doc delta overlays.
#[derive(Default)]
pub struct DocTokenStore {
    base: PackedDocTokens,
    delta: HashMap<u64, Box<[u32]>>,
    deleted: HashMap<u64, ()>,
}

impl DocTokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_packed(base: PackedDocTokens) -> Self {
        Self {
            base,
            delta: HashMap::new(),
            deleted: HashMap::new(),
        }
    }

    pub fn upsert(&mut self, doc_id: u64, tokens: Box<[u32]>) {
        self.deleted.remove(&doc_id);
        self.delta.insert(doc_id, tokens);
    }

    pub fn remove(&mut self, doc_id: u64) {
        self.delta.remove(&doc_id);
        self.deleted.insert(doc_id, ());
    }

    pub fn get(&self, doc_id: u64) -> Option<&[u32]> {
        if self.deleted.contains_key(&doc_id) {
            return None;
        }
        if let Some(tokens) = self.delta.get(&doc_id) {
            return Some(tokens.as_ref());
        }
        self.base.get(doc_id)
    }

    pub fn live_doc_count(&self) -> u64 {
        let mut count = 0u64;
        for &doc_id in &self.base.doc_ids {
            if !self.deleted.contains_key(&doc_id) && !self.delta.contains_key(&doc_id) {
                count += 1;
            }
        }
        count += self.delta.len() as u64;
        count
    }

    pub fn live_term_total(&self) -> u64 {
        let mut total = 0u64;
        for &doc_id in &self.base.doc_ids {
            if self.deleted.contains_key(&doc_id) || self.delta.contains_key(&doc_id) {
                continue;
            }
            if let Some(tokens) = self.base.get(doc_id) {
                total += tokens.iter().filter(|&&t| t != 0).count() as u64;
            }
        }
        for tokens in self.delta.values() {
            total += tokens.iter().filter(|&&t| t != 0).count() as u64;
        }
        total
    }

    /// Fold delta into a new packed base (for persistence).
    pub fn compact_to_packed(&self) -> PackedDocTokens {
        let mut records: Vec<(u64, Box<[u32]>)> = Vec::new();
        for &doc_id in &self.base.doc_ids {
            if self.deleted.contains_key(&doc_id) {
                continue;
            }
            if let Some(tokens) = self.delta.get(&doc_id) {
                records.push((doc_id, tokens.clone()));
            } else if let Some(tokens) = self.base.get(doc_id) {
                records.push((doc_id, tokens.to_vec().into_boxed_slice()));
            }
        }
        for (&doc_id, tokens) in &self.delta {
            if self.base.doc_ids.binary_search(&doc_id).is_err() {
                records.push((doc_id, tokens.clone()));
            }
        }
        PackedDocTokens::from_records(records)
    }

    pub fn replace_base(&mut self, base: PackedDocTokens) {
        self.base = base;
        self.delta.clear();
        self.deleted.clear();
    }
}

/// BM25 ranked hit.
#[derive(Clone, Copy, Debug)]
pub struct RankedHit {
    pub doc_id: u64,
    pub score: f32,
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_roundtrip() {
        let packed = PackedDocTokens::from_records(vec![
            (1, vec![1, 2, 0, 3].into_boxed_slice()),
            (5, vec![2, 2].into_boxed_slice()),
        ]);
        assert_eq!(packed.get(1), Some(&[1, 2, 0, 3][..]));
        let bytes = packed.to_bytes();
        let restored = PackedDocTokens::from_bytes(&bytes).unwrap();
        assert_eq!(restored.get(5), Some(&[2, 2][..]));
    }
}
