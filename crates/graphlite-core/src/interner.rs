use std::collections::HashMap;

use crate::id::StringId;

/// An in-memory String Interner for deduplicating strings across the graph.
///
/// Ensures that repeated strings (such as entity types, relation names, and common labels)
/// are allocated in memory only once and represented by compact 4-byte `StringId` handles.
#[derive(Debug, Clone, Default)]
pub struct StringInterner {
    /// Maps a string to its assigned `StringId`.
    map: HashMap<String, StringId>,
    /// Sequential storage of unique interned strings for O(1) index-based resolution.
    strings: Vec<String>,
    /// Total raw UTF-8 byte count of all unique strings stored.
    total_bytes: usize,
}

impl StringInterner {
    /// Creates a new, empty `StringInterner`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new `StringInterner` pre-allocated with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            strings: Vec::with_capacity(capacity),
            total_bytes: 0,
        }
    }

    /// Interns a string slice, returning its unique `StringId`.
    ///
    /// If the string was already interned, returns the existing `StringId` without any new memory allocation.
    /// If it is a new string, it is stored and assigned the next sequential `StringId`.
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }

        let id = StringId::new(self.strings.len() as u32);
        let owned = s.to_string();
        self.total_bytes += owned.len();
        self.map.insert(owned.clone(), id);
        self.strings.push(owned);
        id
    }

    /// Resolves a `StringId` back to its original string slice.
    ///
    /// Returns `None` if the `StringId` is invalid or does not exist in this interner.
    #[inline]
    pub fn resolve(&self, id: StringId) -> Option<&str> {
        if !id.is_valid() {
            return None;
        }
        self.strings.get(id.as_usize()).map(|s| s.as_str())
    }

    /// Returns the total number of unique strings interned.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if no strings have been interned yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Returns the total byte size of all unique UTF-8 strings stored.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Clears all interned strings and resets the interner.
    pub fn clear(&mut self) {
        self.map.clear();
        self.strings.clear();
        self.total_bytes = 0;
    }

    /// Returns an iterator over all interned `(StringId, &str)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (StringId, &str)> {
        self.strings
            .iter()
            .enumerate()
            .map(|(idx, s)| (StringId::new(idx as u32), s.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_and_resolve() {
        let mut interner = StringInterner::new();

        let id_rust1 = interner.intern("Rust");
        let id_python = interner.intern("Python");
        let id_rust2 = interner.intern("Rust");

        // Duplicate string must return the exact same ID
        assert_eq!(id_rust1, id_rust2);
        // Different string must return a different ID
        assert_ne!(id_rust1, id_python);

        // Resolution
        assert_eq!(interner.resolve(id_rust1), Some("Rust"));
        assert_eq!(interner.resolve(id_python), Some("Python"));
        assert_eq!(interner.resolve(StringId::INVALID), None);
        assert_eq!(interner.resolve(StringId::new(9999)), None);

        // Count
        assert_eq!(interner.len(), 2);
        assert!(!interner.is_empty());
        assert_eq!(interner.total_bytes(), "Rust".len() + "Python".len());
    }

    #[test]
    fn test_with_capacity_and_clear() {
        let mut interner = StringInterner::with_capacity(10);
        assert_eq!(interner.len(), 0);

        interner.intern("Alpha");
        interner.intern("Beta");
        assert_eq!(interner.len(), 2);

        interner.clear();
        assert_eq!(interner.len(), 0);
        assert!(interner.is_empty());
        assert_eq!(interner.total_bytes(), 0);
    }

    #[test]
    fn test_iter() {
        let mut interner = StringInterner::new();
        interner.intern("NodeA");
        interner.intern("NodeB");

        let items: Vec<(StringId, &str)> = interner.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (StringId::new(0), "NodeA"));
        assert_eq!(items[1], (StringId::new(1), "NodeB"));
    }
}
