//! High-Performance Roaring Treemap Operations Module
//!
//! Wraps `roaring::RoaringTreemap` for u64 document IDs.

use roaring::RoaringTreemap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitmapError {
    #[error("Serialization error: {0}")]
    SerializeError(String),
    #[error("Deserialization error: {0}")]
    DeserializeError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// High-performance Bitmap wrapper over RoaringTreemap (u64 doc IDs).
#[derive(Clone, Default)]
pub struct FastBitmap {
    inner: RoaringTreemap,
}

impl FastBitmap {
    /// Create empty bitmap
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: RoaringTreemap::new(),
        }
    }

    /// Create from iterator
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Self {
            inner: RoaringTreemap::from_iter(iter),
        }
    }

    /// Add document ID
    #[inline]
    pub fn add(&mut self, doc_id: u64) -> bool {
        self.inner.insert(doc_id)
    }

    /// Batch add document IDs
    #[inline]
    pub fn add_many(&mut self, doc_ids: &[u64]) {
        self.inner.extend(doc_ids.iter().copied());
    }

    /// Batch add **strictly increasing** document IDs (faster append path).
    /// Falls back to [`Self::add_many`] if the slice is not appendable onto `self`.
    #[inline]
    pub fn add_many_sorted(&mut self, doc_ids: &[u64]) {
        if doc_ids.is_empty() {
            return;
        }
        if self.inner.append(doc_ids.iter().copied()).is_err() {
            self.add_many(doc_ids);
        }
    }

    /// Build from a strictly increasing, duplicate-free slice of document IDs.
    #[inline]
    pub fn from_sorted_slice(doc_ids: &[u64]) -> Self {
        let mut bm = Self::new();
        if !doc_ids.is_empty() {
            let _ = bm.inner.append(doc_ids.iter().copied());
        }
        bm
    }

    /// Remove document ID
    #[inline]
    pub fn remove(&mut self, doc_id: u64) -> bool {
        self.inner.remove(doc_id)
    }

    /// Batch remove document IDs
    #[inline]
    pub fn remove_many(&mut self, doc_ids: &[u64]) {
        for &doc_id in doc_ids {
            self.inner.remove(doc_id);
        }
    }

    /// Check if contains document ID
    #[inline]
    pub fn contains(&self, doc_id: u64) -> bool {
        self.inner.contains(doc_id)
    }

    /// Get document count
    #[inline]
    pub fn len(&self) -> u64 {
        self.inner.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Intersection operation (in-place)
    #[inline]
    pub fn and_inplace(&mut self, other: &FastBitmap) {
        self.inner &= &other.inner;
    }

    /// Intersection operation (returns new bitmap)
    #[inline]
    pub fn and(&self, other: &FastBitmap) -> FastBitmap {
        FastBitmap {
            inner: &self.inner & &other.inner,
        }
    }

    /// Union operation (in-place)
    #[inline]
    pub fn or_inplace(&mut self, other: &FastBitmap) {
        self.inner |= &other.inner;
    }

    /// Union operation (returns new bitmap)
    #[inline]
    pub fn or(&self, other: &FastBitmap) -> FastBitmap {
        FastBitmap {
            inner: &self.inner | &other.inner,
        }
    }

    /// Difference operation
    #[inline]
    pub fn andnot(&self, other: &FastBitmap) -> FastBitmap {
        FastBitmap {
            inner: &self.inner - &other.inner,
        }
    }

    /// Difference operation (in-place)
    #[inline]
    pub fn andnot_inplace(&mut self, other: &FastBitmap) {
        self.inner -= &other.inner;
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Result<Vec<u8>, BitmapError> {
        let mut bytes = Vec::new();
        self.inner
            .serialize_into(&mut bytes)
            .map_err(|e| BitmapError::SerializeError(e.to_string()))?;
        Ok(bytes)
    }

    /// Deserialize from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, BitmapError> {
        let inner = RoaringTreemap::deserialize_from(bytes)
            .map_err(|e| BitmapError::DeserializeError(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get iterator
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.inner.iter()
    }

    /// Convert to Vec
    #[inline]
    pub fn to_vec(&self) -> Vec<u64> {
        self.inner.iter().collect()
    }

    /// Get reference to inner bitmap
    #[inline]
    pub fn inner(&self) -> &RoaringTreemap {
        &self.inner
    }

    /// Get mutable reference to inner bitmap
    #[inline]
    pub fn inner_mut(&mut self) -> &mut RoaringTreemap {
        &mut self.inner
    }

    /// Take ownership of the inner treemap
    #[inline]
    pub fn into_inner(self) -> RoaringTreemap {
        self.inner
    }
}

/// Batch intersection: sort by size, start from smallest
#[inline]
pub fn fast_intersection(bitmaps: &[&FastBitmap]) -> FastBitmap {
    if bitmaps.is_empty() {
        return FastBitmap::new();
    }

    if bitmaps.len() == 1 {
        return bitmaps[0].clone();
    }

    let mut sorted: Vec<_> = bitmaps.iter().collect();
    sorted.sort_by_key(|b| b.len());

    let mut result = (*sorted[0]).clone();

    for bitmap in sorted.iter().skip(1) {
        result.and_inplace(bitmap);
        if result.is_empty() {
            return result;
        }
    }

    result
}

/// Batch union operation
#[inline]
pub fn fast_union(bitmaps: &[&FastBitmap]) -> FastBitmap {
    if bitmaps.is_empty() {
        return FastBitmap::new();
    }

    if bitmaps.len() == 1 {
        return bitmaps[0].clone();
    }

    let mut result = FastBitmap::new();
    for bitmap in bitmaps {
        result.or_inplace(bitmap);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut bitmap = FastBitmap::new();
        bitmap.add(1);
        bitmap.add(2);
        bitmap.add(3);

        assert_eq!(bitmap.len(), 3);
        assert!(bitmap.contains(1));
        assert!(!bitmap.contains(4));
    }

    #[test]
    fn test_u64_ids() {
        let large = u32::MAX as u64 + 17;
        let mut bitmap = FastBitmap::new();
        bitmap.add(large);
        assert!(bitmap.contains(large));
        assert_eq!(bitmap.to_vec(), vec![large]);
    }

    #[test]
    fn test_intersection() {
        let bitmap1 = FastBitmap::from_iter([1, 2, 3, 4, 5]);
        let bitmap2 = FastBitmap::from_iter([3, 4, 5, 6, 7]);

        let result = bitmap1.and(&bitmap2);
        assert_eq!(result.len(), 3);
        assert!(result.contains(3));
        assert!(result.contains(4));
        assert!(result.contains(5));
    }

    #[test]
    fn test_serialization() {
        let bitmap = FastBitmap::from_iter([1, 100, 1000, 10000, u32::MAX as u64 + 1]);
        let bytes = bitmap.serialize().unwrap();
        let restored = FastBitmap::deserialize(&bytes).unwrap();

        assert_eq!(bitmap.len(), restored.len());
        for id in bitmap.iter() {
            assert!(restored.contains(id));
        }
    }
}
