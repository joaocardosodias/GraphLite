use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::tempdir;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::graph::adjacency::AdjacencyGraph;
use graphlite_core::graph::traversal::{bfs_csr, TraversalConfig};
use graphlite_core::id::{EdgeId, NodeId, StringId};
use graphlite_core::record::{EdgeRecord, NodeRecord};
use graphlite_core::vector::distance::{cosine_similarity, Metric};
use graphlite_core::vector::quantization::{Quantization, QuantizedVector};
use graphlite_core::vector::simd::simd_cosine_similarity;

fn bench_simd_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_cosine");

    for &dim in &[384, 768, 1536] {
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 + 0.1).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 - 0.05).collect();

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |b, _| {
            b.iter(|| cosine_similarity(black_box(&v1), black_box(&v2)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| simd_cosine_similarity(black_box(&v1), black_box(&v2)).unwrap());
        });
    }

    group.finish();
}

fn bench_quantized_sq8_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_quantized_vs_float32");
    let dim = 384;

    let v1: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 + 0.1).collect();
    let v2: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 - 0.05).collect();
    let q2 = QuantizedVector::quantize(&v2);

    group.bench_function("float32_simd_cosine", |b| {
        b.iter(|| simd_cosine_similarity(black_box(&v1), black_box(&v2)).unwrap());
    });

    group.bench_function("sq8_asymmetric_dot_product", |b| {
        b.iter(|| q2.dot_product_asymmetric(black_box(&v1)).unwrap());
    });

    group.finish();
}

fn bench_csr_graph_bfs_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_bfs_traversal");

    let num_nodes = 5_000;
    let mut adj = AdjacencyGraph::new();

    for i in 0..num_nodes {
        let nid = NodeId::new(i);
        adj.add_node(NodeRecord::new(nid, StringId::new(i + 1), StringId::INVALID, StringId::INVALID, 0)).unwrap();
    }

    let mut edge_counter = 1;
    for i in 0..num_nodes {
        let src = NodeId::new(i);
        let tgt1 = NodeId::new((i + 1) % num_nodes);
        let tgt2 = NodeId::new((i + 7) % num_nodes);
        let tgt3 = NodeId::new((i + 31) % num_nodes);

        adj.add_edge(EdgeRecord::new(EdgeId::new(edge_counter), src, tgt1, StringId::new(1))).unwrap();
        edge_counter += 1;
        adj.add_edge(EdgeRecord::new(EdgeId::new(edge_counter), src, tgt2, StringId::new(1))).unwrap();
        edge_counter += 1;
        adj.add_edge(EdgeRecord::new(EdgeId::new(edge_counter), src, tgt3, StringId::new(1))).unwrap();
        edge_counter += 1;
    }

    let csr = adj.to_csr();
    let seeds = vec![NodeId::new(0), NodeId::new(100), NodeId::new(500)];

    for &depth in &[1, 2, 3] {
        let config = TraversalConfig {
            max_depth: depth,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("csr_bfs_depth", depth), &depth, |b, _| {
            b.iter(|| bfs_csr(black_box(&csr), black_box(&seeds), black_box(&config)));
        });
    }

    group.finish();
}

fn bench_end_to_end_retrieve_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_retrieve_context");

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench_db.graph");

    let config = GraphLiteConfig::new()
        .with_dim(384)
        .with_metric(Metric::Cosine)
        .with_quantization(Quantization::ScalarInt8)
        .with_max_tokens(1500)
        .with_auto_flush(false);

    let engine = GraphLiteEngine::open_or_create(&db_path, config).unwrap();

    // Populate 200 nodes with edges
    for i in 0..200 {
        let name = format!("Entity_{}", i);
        let v: Vec<f32> = (0..384).map(|d| ((i * 384 + d) as f32 * 0.001).sin()).collect();
        let nid = engine.upsert_node(&name, "Concept", "Benchmark Entity Description", Some(&v)).unwrap();

        if i > 0 {
            let prev = NodeId::new((i - 1) as u32);
            let _ = engine.add_edge(nid, prev, "CONNECTS", 0.9, true);
        }
    }
    engine.flush().unwrap();

    let query_vector: Vec<f32> = (0..384).map(|d| (d as f32 * 0.005).cos()).collect();

    group.bench_function("retrieve_context_200_nodes_384d", |b| {
        b.iter(|| {
            engine.retrieve_context(black_box(&query_vector), None).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_cosine_similarity,
    bench_quantized_sq8_dot_product,
    bench_csr_graph_bfs_traversal,
    bench_end_to_end_retrieve_context
);
criterion_main!(benches);
