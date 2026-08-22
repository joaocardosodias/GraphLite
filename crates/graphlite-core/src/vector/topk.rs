use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Internal wrapper pairing an item with its relevance score for Min-Heap ordering.
#[derive(Debug, Clone)]
struct MinScoredItem<T> {
    item: T,
    score: f32,
}

impl<T> PartialEq for MinScoredItem<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for MinScoredItem<T> {}

impl<T> PartialOrd for MinScoredItem<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for MinScoredItem<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order so that BinaryHeap acts as a MIN-HEAP (smallest score on top)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// A bounded Min-Heap priority queue for efficiently extracting the Top-K highest-scoring items.
///
/// Ensures $O(N \log K)$ time complexity and strictly bounded $O(K)$ memory footprint,
/// discarding inferior candidates in $O(1)$ time without requiring large memory allocations.
#[derive(Debug, Clone)]
pub struct TopKQueue<T> {
    capacity: usize,
    heap: BinaryHeap<MinScoredItem<T>>,
}

impl<T> TopKQueue<T> {
    /// Creates a new `TopKQueue` with a fixed maximum capacity $K$.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Pushes an item with its associated score into the queue.
    ///
    /// If the queue is below capacity, the item is inserted directly.
    /// If the queue is at capacity and `score` is higher than the current minimum,
    /// the minimum item is replaced. Otherwise, the candidate is discarded in $O(1)$.
    #[inline]
    pub fn push(&mut self, item: T, score: f32) {
        if self.capacity == 0 {
            return;
        }

        if self.heap.len() < self.capacity {
            self.heap.push(MinScoredItem { item, score });
        } else if let Some(min) = self.heap.peek() {
            if score > min.score {
                self.heap.pop();
                self.heap.push(MinScoredItem { item, score });
            }
        }
    }

    /// Returns the minimum score currently held in the Top-K queue.
    ///
    /// Useful for early-exit filtering during search loops.
    #[inline]
    pub fn min_score(&self) -> Option<f32> {
        self.heap.peek().map(|item| item.score)
    }

    /// Consumes the queue and returns all Top-K items sorted in **descending** order (highest score first).
    pub fn into_sorted_vec(self) -> Vec<(T, f32)> {
        let mut vec: Vec<(T, f32)> = self
            .heap
            .into_iter()
            .map(|item| (item.item, item.score))
            .collect();

        // Sort descending by score
        vec.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        vec
    }

    /// Returns the current number of elements in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the maximum capacity $K$ of this queue.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears all elements from the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_selection_and_ordering() {
        let mut queue = TopKQueue::new(3);

        queue.push("D", 0.4);
        queue.push("A", 0.9);
        queue.push("B", 0.8);
        queue.push("E", 0.2);
        queue.push("C", 0.7);

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.min_score(), Some(0.7));

        let results = queue.into_sorted_vec();
        assert_eq!(results.len(), 3);

        // Must contain top 3: A (0.9), B (0.8), C (0.7) in descending order
        assert_eq!(results[0], ("A", 0.9));
        assert_eq!(results[1], ("B", 0.8));
        assert_eq!(results[2], ("C", 0.7));
    }

    #[test]
    fn test_empty_and_zero_capacity() {
        let mut zero_queue: TopKQueue<i32> = TopKQueue::new(0);
        zero_queue.push(1, 10.0);
        assert_eq!(zero_queue.len(), 0);
        assert!(zero_queue.is_empty());
        assert_eq!(zero_queue.into_sorted_vec(), vec![]);

        let empty_queue: TopKQueue<&str> = TopKQueue::new(5);
        assert_eq!(empty_queue.into_sorted_vec(), vec![]);
    }

    #[test]
    fn test_fewer_elements_than_capacity() {
        let mut queue = TopKQueue::new(5);
        queue.push(10, 0.5);
        queue.push(20, 0.95);

        let results = queue.into_sorted_vec();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (20, 0.95));
        assert_eq!(results[1], (10, 0.5));
    }
}
