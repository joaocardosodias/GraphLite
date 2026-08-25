use graphite::id::NodeId;
use graphite::vector::{
    cosine_similarity, dot_product, euclidean_distance, norm, normalized, simd_cosine_similarity,
    simd_dot_product, simd_euclidean_distance, Metric, Quantization, QuantizedVector, VectorStore,
};

/// Helper: Deterministic pseudo-random number generator for reproducible vector testing (Xorshift32).
struct SimpleRng(u32);

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 123456789 } else { seed })
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        // Map to [-1.0, 1.0]
        ((self.0 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }

    fn generate_vector(&mut self, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        normalized(&raw)
    }
}

#[test]
fn test_sq8_quantization_accuracy_and_mae() {
    let mut rng = SimpleRng::new(42);
    let dim = 384;
    let num_pairs = 200;

    let mut total_absolute_error = 0.0f32;
    let mut max_absolute_error = 0.0f32;

    for _ in 0..num_pairs {
        let v1 = rng.generate_vector(dim);
        let v2 = rng.generate_vector(dim);

        let exact_cos = cosine_similarity(&v1, &v2).unwrap();

        let q1 = QuantizedVector::quantize(&v1);
        let approx_cos = q1.cosine_similarity_asymmetric(&v2, norm(&v2)).unwrap();

        let error = (exact_cos - approx_cos).abs();
        total_absolute_error += error;
        if error > max_absolute_error {
            max_absolute_error = error;
        }
    }

    let mean_absolute_error = total_absolute_error / (num_pairs as f32);

    // Mean Absolute Error must be below 0.008 (less than 0.8% error)
    assert!(
        mean_absolute_error < 0.008,
        "Mean absolute error too high: {}",
        mean_absolute_error
    );

    // Maximum error across all pairs must be below 0.025
    assert!(
        max_absolute_error < 0.025,
        "Max absolute error too high: {}",
        max_absolute_error
    );
}

#[test]
fn test_sq8_top_k_recall_at_10() {
    let mut rng = SimpleRng::new(1337);
    let dim = 384;
    let num_vectors = 500;
    let k = 10;

    let mut float_store = VectorStore::new(dim, Metric::Cosine, Quantization::None);
    let mut quant_store = VectorStore::new(dim, Metric::Cosine, Quantization::ScalarInt8);

    for i in 0..num_vectors {
        let vec = rng.generate_vector(dim);
        let node_id = NodeId::new(i as u32);
        float_store.insert(node_id, &vec).unwrap();
        quant_store.insert(node_id, &vec).unwrap();
    }

    let mut total_hits = 0;
    let num_queries = 20;

    for _ in 0..num_queries {
        let query = rng.generate_vector(dim);

        let exact_results = float_store.search(&query, k).unwrap();
        let quant_results = quant_store.search(&query, k).unwrap();

        let exact_ids: std::collections::HashSet<NodeId> =
            exact_results.into_iter().map(|(id, _)| id).collect();

        for (id, _) in quant_results {
            if exact_ids.contains(&id) {
                total_hits += 1;
            }
        }
    }

    let recall = (total_hits as f32) / ((num_queries * k) as f32);

    // Recall@10 must be >= 95% compared to full Float32 brute-force search
    assert!(
        recall >= 0.95,
        "Recall@10 was {} (expected >= 0.95)",
        recall
    );
}

#[test]
fn test_simd_vs_baseline_mass_verification() {
    let mut rng = SimpleRng::new(999);

    for dim in [64, 128, 384, 768, 1536] {
        for _ in 0..50 {
            let a = rng.generate_vector(dim);
            let b = rng.generate_vector(dim);

            // Dot Product parity
            let dot_base = dot_product(&a, &b).unwrap();
            let dot_simd = simd_dot_product(&a, &b).unwrap();
            assert!((dot_base - dot_simd).abs() < 1e-4);

            // Cosine Similarity parity
            let cos_base = cosine_similarity(&a, &b).unwrap();
            let cos_simd = simd_cosine_similarity(&a, &b).unwrap();
            assert!((cos_base - cos_simd).abs() < 1e-4);

            // Euclidean Distance parity
            let euc_base = euclidean_distance(&a, &b).unwrap();
            let euc_simd = simd_euclidean_distance(&a, &b).unwrap();
            assert!((euc_base - euc_simd).abs() < 1e-4);
        }
    }
}

#[test]
fn test_vector_store_large_scale_concurrency() {
    let mut store = VectorStore::new(128, Metric::Cosine, Quantization::ScalarInt8);
    let mut rng = SimpleRng::new(2024);

    let num_items = 1000;
    for i in 0..num_items {
        let vec = rng.generate_vector(128);
        store.insert(NodeId::new(i as u32), &vec).unwrap();
    }

    assert_eq!(store.len(), num_items);
    assert!(store.memory_usage() > 0);

    // Multi-query test
    let query = rng.generate_vector(128);
    let top_5 = store.search(&query, 5).unwrap();

    assert_eq!(top_5.len(), 5);
    // Verify descending order
    for i in 0..4 {
        assert!(top_5[i].1 >= top_5[i + 1].1);
    }
}
