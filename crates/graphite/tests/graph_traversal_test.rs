use graphite::graph::{
    bfs_adjacency, bfs_csr, extract_subgraph_adjacency, extract_subgraph_csr, AdjacencyGraph,
    CsrGraph, HybridScoreConfig, TraversalConfig, TraversalDirection,
};
use graphite::id::{EdgeId, NodeId, StringId};
use graphite::record::{EdgeRecord, NodeRecord, NO_VECTOR_OFFSET};

/// Helper to create a NodeRecord with minimal boilerplate.
fn make_node(id: u32, name_id: u32) -> NodeRecord {
    NodeRecord::new(
        NodeId::new(id),
        StringId::new(name_id),
        StringId::new(1),
        StringId::INVALID,
        NO_VECTOR_OFFSET,
    )
}

/// Helper to create an EdgeRecord with minimal boilerplate.
fn make_edge(id: u32, source: u32, target: u32, weight: f32) -> EdgeRecord {
    EdgeRecord::new(
        EdgeId::new(id),
        NodeId::new(source),
        NodeId::new(target),
        StringId::new(10),
    )
    .with_weight(weight)
}

#[test]
fn test_complex_cyclic_graph_traversal() {
    let mut graph = AdjacencyGraph::new();

    // Create 6 nodes: 0, 1, 2, 3, 4, 5
    for i in 0..6 {
        graph.add_node(make_node(i, i * 10)).unwrap();
    }

    // Cycle 1: 0 -> 1 -> 2 -> 0
    // Bridge: 2 -> 3
    // Cycle 2: 3 -> 4 -> 5 -> 3
    graph.add_edge(make_edge(1, 0, 1, 0.9)).unwrap();
    graph.add_edge(make_edge(2, 1, 2, 0.9)).unwrap();
    graph.add_edge(make_edge(3, 2, 0, 0.9)).unwrap(); // Cycle back to 0
    graph.add_edge(make_edge(4, 2, 3, 0.8)).unwrap(); // Bridge to cycle 2
    graph.add_edge(make_edge(5, 3, 4, 0.7)).unwrap();
    graph.add_edge(make_edge(6, 4, 5, 0.7)).unwrap();
    graph.add_edge(make_edge(7, 5, 3, 0.7)).unwrap(); // Cycle back to 3

    let csr: CsrGraph = (&graph).into();

    let config = TraversalConfig {
        max_depth: 3,
        min_edge_weight: 0.5,
        max_nodes: 50,
        direction: TraversalDirection::Outgoing,
    };

    let traversed = bfs_csr(&csr, &[NodeId::new(0)], &config);

    // Nodes reachable within 3 hops from 0:
    // Depth 0: 0
    // Depth 1: 1
    // Depth 2: 2
    // Depth 3: 3 (reached via 2->3; 0 is NOT re-added because of cycle detection)
    let visited_ids: Vec<u32> = traversed.iter().map(|n| n.node_id.as_u32()).collect();
    assert_eq!(visited_ids, vec![0, 1, 2, 3]);

    // Ensure depths are exact
    assert_eq!(traversed[0].depth, 0);
    assert_eq!(traversed[1].depth, 1);
    assert_eq!(traversed[2].depth, 2);
    assert_eq!(traversed[3].depth, 3);
}

#[test]
fn test_disconnected_islands_isolation() {
    let mut graph = AdjacencyGraph::new();

    // Island 1: Nodes 0, 1
    // Island 2: Nodes 10, 11
    graph.add_node(make_node(0, 1)).unwrap();
    graph.add_node(make_node(1, 2)).unwrap();
    graph.add_node(make_node(10, 3)).unwrap();
    graph.add_node(make_node(11, 4)).unwrap();

    graph.add_edge(make_edge(1, 0, 1, 0.95)).unwrap();
    graph.add_edge(make_edge(2, 10, 11, 0.95)).unwrap();

    let csr: CsrGraph = (&graph).into();
    let config = TraversalConfig::default();

    // Search starting only in Island 1 (Node 0)
    let traversed = bfs_csr(&csr, &[NodeId::new(0)], &config);
    let ids: Vec<u32> = traversed.iter().map(|n| n.node_id.as_u32()).collect();

    assert_eq!(ids, vec![0, 1]);
    assert!(!ids.contains(&10));
    assert!(!ids.contains(&11));
}

#[test]
fn test_adjacency_and_csr_traversal_parity() {
    let mut graph = AdjacencyGraph::new();

    // Build a multi-branch tree:
    // 0 -> 1 (0.9), 0 -> 2 (0.85), 1 -> 3 (0.8), 1 -> 4 (0.75), 2 -> 5 (0.7)
    for i in 0..6 {
        graph.add_node(make_node(i, i)).unwrap();
    }
    graph.add_edge(make_edge(1, 0, 1, 0.90)).unwrap();
    graph.add_edge(make_edge(2, 0, 2, 0.85)).unwrap();
    graph.add_edge(make_edge(3, 1, 3, 0.80)).unwrap();
    graph.add_edge(make_edge(4, 1, 4, 0.75)).unwrap();
    graph.add_edge(make_edge(5, 2, 5, 0.70)).unwrap();

    let csr: CsrGraph = (&graph).into();

    let config = TraversalConfig {
        max_depth: 2,
        min_edge_weight: 0.5,
        max_nodes: 20,
        direction: TraversalDirection::Outgoing,
    };

    let seeds = [NodeId::new(0)];
    let adj_traversed = bfs_adjacency(&graph, &seeds, &config);
    let csr_traversed = bfs_csr(&csr, &seeds, &config);

    assert_eq!(adj_traversed.len(), csr_traversed.len());

    for (adj_node, csr_node) in adj_traversed.iter().zip(csr_traversed.iter()) {
        assert_eq!(adj_node.node_id, csr_node.node_id);
        assert_eq!(adj_node.depth, csr_node.depth);
        assert!((adj_node.path_weight - csr_node.path_weight).abs() < 1e-4);
    }
}

#[test]
fn test_end_to_end_subgraph_extraction_with_lateral_cross_edges() {
    let mut graph = AdjacencyGraph::new();

    // Triangle with cross edge:
    // 0 -> 1 (0.9)
    // 1 -> 2 (0.8)
    // 0 -> 2 (0.95 - lateral direct connection!)
    // 3 is an isolated node
    for i in 0..4 {
        graph.add_node(make_node(i, i * 5)).unwrap();
    }
    graph.add_edge(make_edge(10, 0, 1, 0.90)).unwrap();
    graph.add_edge(make_edge(20, 1, 2, 0.80)).unwrap();
    graph.add_edge(make_edge(30, 0, 2, 0.95)).unwrap();

    let csr: CsrGraph = (&graph).into();

    let seeds = vec![(NodeId::new(0), 0.95)];
    let t_config = TraversalConfig {
        max_depth: 2,
        min_edge_weight: 0.5,
        ..Default::default()
    };
    let h_config = HybridScoreConfig {
        alpha: 0.6,
        depth_decay: 0.85,
        min_score_threshold: 0.05,
        relative_drop_off: None,
        ..Default::default()
    };

    let subgraph_csr = extract_subgraph_csr(&csr, &seeds, &t_config, &h_config);
    let subgraph_adj = extract_subgraph_adjacency(&graph, &seeds, &t_config, &h_config);

    assert_eq!(subgraph_csr.entity_count(), 3);
    assert_eq!(subgraph_csr.edge_count(), 3);
    assert_eq!(subgraph_adj.entity_count(), 3);
    assert_eq!(subgraph_adj.edge_count(), 3);

    // Cross-edge 0->2 (Edge 30) must be captured
    assert!(subgraph_csr.edges.iter().any(|e| e.id == EdgeId::new(30)));
    assert!(subgraph_adj.edges.iter().any(|e| e.id == EdgeId::new(30)));

    // Entities must be sorted by final score descending
    assert!(subgraph_csr.entities[0].final_score >= subgraph_csr.entities[1].final_score);
    assert!(subgraph_csr.entities[1].final_score >= subgraph_csr.entities[2].final_score);
}
