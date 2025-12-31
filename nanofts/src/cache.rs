//! Lock-Free Concurrent Cache Module
//! 
//! Provides high-performance LRU cache implementation with concurrent access support

use dashmap::DashMap;
use parking_lot::RwLock;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use ahash::RandomState;

/// Cache entry
struct CacheEntry<V> {
    value: V,
    access_count: AtomicU64,
    last_access: AtomicU64,
}

impl<V> CacheEntry<V> {
    fn new(value: V) -> Self {
        Self {
            value,
            access_count: AtomicU64::new(1),
            last_access: AtomicU64::new(current_timestamp()),
        }
    }

    fn touch(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access.store(current_timestamp(), Ordering::Relaxed);
    }
}

/// Get current timestamp (nanoseconds)
#[inline]
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// High-performance concurrent LRU cache
/// 
/// Uses DashMap for lock-free concurrent access
pub struct ConcurrentLruCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    map: DashMap<K, Arc<CacheEntry<V>>, RandomState>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<K, V> ConcurrentLruCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new cache
    pub fn new(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity_and_hasher(capacity, RandomState::new()),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get cached value
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(entry) = self.map.get(key) {
            entry.touch();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert cached value
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        // Check if eviction needed
        if self.map.len() >= self.capacity {
            self.evict();
        }
        
        let entry = Arc::new(CacheEntry::new(value));
        self.map.insert(key, entry);
    }

    /// Get or insert
    #[inline]
    pub fn get_or_insert<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce() -> V,
    {
        // Try to get first
        if let Some(entry) = self.map.get(&key) {
            entry.touch();
            self.hits.fetch_add(1, Ordering::Relaxed);
            return entry.value.clone();
        }

        // Cache miss, compute value
        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = f();
        
        // Check if eviction needed
        if self.map.len() >= self.capacity {
            self.evict();
        }
        
        let entry = Arc::new(CacheEntry::new(value.clone()));
        self.map.insert(key, entry);
        
        value
    }

    /// Remove cache entry
    #[inline]
    pub fn remove(&self, key: &K) -> Option<V> {
        self.map.remove(key).map(|(_, entry)| entry.value.clone())
    }

    /// Clear cache
    pub fn clear(&self) {
        self.map.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get cache size
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    /// Evict least used entries
    fn evict(&self) {
        // Find entries with least access count for eviction
        let target_remove = self.capacity / 4; // Remove 25%
        
        let mut entries: Vec<_> = self.map.iter()
            .map(|entry| {
                let key = entry.key().clone();
                let score = entry.value().access_count.load(Ordering::Relaxed);
                (key, score)
            })
            .collect();
        
        // Sort by access count
        entries.sort_by_key(|(_, score)| *score);
        
        // Remove entries with least access count
        for (key, _) in entries.into_iter().take(target_remove) {
            self.map.remove(&key);
        }
    }
}

/// Sharded cache - for ultra-large scale data
/// 
/// Splits cache into multiple shards to reduce lock contention
pub struct ShardedCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    shards: Vec<ConcurrentLruCache<K, V>>,
    shard_count: usize,
}

impl<K, V> ShardedCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create sharded cache
    pub fn new(total_capacity: usize, shard_count: usize) -> Self {
        let shard_capacity = total_capacity / shard_count;
        let shards: Vec<_> = (0..shard_count)
            .map(|_| ConcurrentLruCache::new(shard_capacity))
            .collect();
        
        Self {
            shards,
            shard_count,
        }
    }

    /// Get shard index
    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        use std::hash::Hasher;
        let mut hasher = ahash::AHasher::default();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.shard_count
    }

    /// Get cached value
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        self.shards[idx].get(key)
    }

    /// Insert cached value
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        let idx = self.shard_index(&key);
        self.shards[idx].insert(key, value);
    }

    /// Get or insert
    #[inline]
    pub fn get_or_insert<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce() -> V,
    {
        let idx = self.shard_index(&key);
        self.shards[idx].get_or_insert(key, f)
    }

    /// Clear all shards
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
    }

    /// Get total size
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get average hit rate
    pub fn hit_rate(&self) -> f64 {
        let rates: Vec<_> = self.shards.iter().map(|s| s.hit_rate()).collect();
        rates.iter().sum::<f64>() / rates.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_cache() {
        let cache = ConcurrentLruCache::new(100);
        
        cache.insert("key1".to_string(), 1);
        cache.insert("key2".to_string(), 2);
        
        assert_eq!(cache.get(&"key1".to_string()), Some(1));
        assert_eq!(cache.get(&"key2".to_string()), Some(2));
        assert_eq!(cache.get(&"key3".to_string()), None);
    }

    #[test]
    fn test_sharded_cache() {
        let cache = ShardedCache::<String, i32>::new(1000, 16);
        
        for i in 0..100 {
            cache.insert(format!("key{}", i), i);
        }
        
        for i in 0..100 {
            assert_eq!(cache.get(&format!("key{}", i)), Some(i));
        }
    }
}
