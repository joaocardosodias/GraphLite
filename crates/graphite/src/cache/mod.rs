//! High-performance, thread-safe LRU Caching System for Graphite.
//!
//! Provides two cache layers:
//! 1. `EmbeddingCache`: Caches text -> float32 vector embeddings.
//! 2. `QueryCache`: Caches (vector_hash, query_options) -> `RetrievedContext`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::engine::query::RetrievedContext;

/// Statistics tracking cache efficiency and hit rates.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheStats {
    pub capacity: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

struct LruEntry<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A fast, zero-dependency Least-Recently-Used (LRU) cache using a slot-based linked list.
pub struct LruCache<K, V> {
    capacity: usize,
    slots: Vec<Option<LruEntry<K, V>>>,
    free_slots: Vec<usize>,
    lookup: HashMap<K, usize>,
    head: Option<usize>, // Most recently used
    tail: Option<usize>, // Least recently used
    hits: u64,
    misses: u64,
}

impl<K: Clone + Eq + Hash, V: Clone> LruCache<K, V> {
    /// Creates a new LRU cache with the specified maximum capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            slots: Vec::with_capacity(cap),
            free_slots: Vec::new(),
            lookup: HashMap::with_capacity(cap),
            head: None,
            tail: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Returns the number of currently stored entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.lookup.len()
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lookup.is_empty()
    }

    /// Clears all entries and resets metrics.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_slots.clear();
        self.lookup.clear();
        self.head = None;
        self.tail = None;
    }

    /// Returns current cache performance metrics.
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            (self.hits as f64) / (total as f64)
        } else {
            0.0
        };
        CacheStats {
            capacity: self.capacity,
            entries: self.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: (hit_rate * 10000.0).round() / 100.0, // percentage e.g. 85.50
        }
    }

    /// Retrieves an entry from the cache, promoting it to most recently used.
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(&idx) = self.lookup.get(key) {
            self.hits += 1;
            self.promote(idx);
            self.slots[idx].as_ref().map(|e| e.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Inserts or updates an entry in the cache.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(&idx) = self.lookup.get(&key) {
            if let Some(entry) = self.slots[idx].as_mut() {
                entry.value = value;
            }
            self.promote(idx);
            return;
        }

        if self.len() >= self.capacity {
            self.evict_lru();
        }

        let slot_idx = if let Some(free_idx) = self.free_slots.pop() {
            self.slots[free_idx] = Some(LruEntry {
                key: key.clone(),
                value,
                prev: None,
                next: self.head,
            });
            free_idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Some(LruEntry {
                key: key.clone(),
                value,
                prev: None,
                next: self.head,
            }));
            idx
        };

        if let Some(old_head) = self.head {
            if let Some(Some(entry)) = self.slots.get_mut(old_head) {
                entry.prev = Some(slot_idx);
            }
        }
        self.head = Some(slot_idx);
        if self.tail.is_none() {
            self.tail = Some(slot_idx);
        }

        self.lookup.insert(key, slot_idx);
    }

    fn promote(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }

        let (prev, next) = match &self.slots[idx] {
            Some(e) => (e.prev, e.next),
            None => return,
        };

        // Detach from current position
        if let Some(p) = prev {
            if let Some(Some(e)) = self.slots.get_mut(p) {
                e.next = next;
            }
        }
        if let Some(n) = next {
            if let Some(Some(e)) = self.slots.get_mut(n) {
                e.prev = prev;
            }
        }
        if self.tail == Some(idx) {
            self.tail = prev;
        }

        // Attach as new head
        if let Some(Some(e)) = self.slots.get_mut(idx) {
            e.prev = None;
            e.next = self.head;
        }
        if let Some(old_head) = self.head {
            if let Some(Some(e)) = self.slots.get_mut(old_head) {
                e.prev = Some(idx);
            }
        }
        self.head = Some(idx);
    }

    fn evict_lru(&mut self) {
        if let Some(tail_idx) = self.tail {
            if let Some(entry) = self.slots[tail_idx].take() {
                self.lookup.remove(&entry.key);
                self.tail = entry.prev;
                if let Some(new_tail) = self.tail {
                    if let Some(Some(e)) = self.slots.get_mut(new_tail) {
                        e.next = None;
                    }
                } else {
                    self.head = None;
                }
                self.free_slots.push(tail_idx);
            }
        }
    }
}

/// Unique cache key for GraphRAG query results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    pub vector_hash: u64,
    pub threshold: Option<u32>,
    pub type_filter: Option<Vec<String>>,
    pub top_k_seeds: usize,
}

impl QueryCacheKey {
    /// Computes a hash key from a query vector and options.
    pub fn new(
        vector: &[f32],
        threshold: Option<f32>,
        type_filter: Option<&[String]>,
        top_k_seeds: usize,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &val in vector {
            val.to_bits().hash(&mut hasher);
        }
        let vector_hash = hasher.finish();

        Self {
            vector_hash,
            threshold: threshold.map(|t| t.to_bits()),
            type_filter: type_filter.map(|f| f.to_vec()),
            top_k_seeds,
        }
    }
}

/// Thread-safe in-memory cache for GraphRAG retrieved contexts.
pub struct QueryCache {
    lru: LruCache<QueryCacheKey, RetrievedContext>,
}

impl QueryCache {
    /// Creates a new query result cache with capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            lru: LruCache::new(capacity),
        }
    }

    /// Looks up a cached query result.
    #[inline]
    pub fn get(&mut self, key: &QueryCacheKey) -> Option<RetrievedContext> {
        self.lru.get(key)
    }

    /// Stores a query result in cache.
    #[inline]
    pub fn insert(&mut self, key: QueryCacheKey, context: RetrievedContext) {
        self.lru.insert(key, context);
    }

    /// Clears the cache on database mutations.
    #[inline]
    pub fn clear(&mut self) {
        self.lru.clear();
    }

    /// Returns cache efficiency statistics.
    #[inline]
    pub fn stats(&self) -> CacheStats {
        self.lru.stats()
    }
}

/// In-memory cache for text embeddings.
pub struct EmbeddingCache {
    lru: LruCache<u64, Vec<f32>>,
}

impl EmbeddingCache {
    /// Creates a new embedding cache with capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            lru: LruCache::new(capacity),
        }
    }

    /// Looks up an embedding by hashing the input text.
    pub fn get(&mut self, text: &str) -> Option<Vec<f32>> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        self.lru.get(&hasher.finish())
    }

    /// Caches an embedding vector for a given text.
    pub fn insert(&mut self, text: &str, vector: Vec<f32>) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        self.lru.insert(hasher.finish(), vector);
    }

    /// Clears the embedding cache.
    #[inline]
    pub fn clear(&mut self) {
        self.lru.clear();
    }

    /// Returns embedding cache statistics.
    #[inline]
    pub fn stats(&self) -> CacheStats {
        self.lru.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.stats().hits, 1);

        // Inserting 'c' should evict 'b' since 'a' was recently accessed
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(10);
        let vec = vec![0.1, 0.2, 0.3];
        cache.insert("reembolso integral", vec.clone());

        assert_eq!(cache.get("reembolso integral"), Some(vec));
        assert_eq!(cache.get("outro texto"), None);
    }
}
