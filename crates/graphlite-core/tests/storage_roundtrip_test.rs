use std::fs::OpenOptions;
use std::io::Write;
use tempfile::tempdir;

use graphlite_core::error::GraphLiteError;
use graphlite_core::graph::csr::CsrGraph;
use graphlite_core::id::{EdgeId, NodeId, StringId};
use graphlite_core::interner::StringInterner;
use graphlite_core::record::{EdgeRecord, NodeRecord, NO_VECTOR_OFFSET};
use graphlite_core::storage::{write_database_atomic, MmapGraphReader, GRAPH_MAGIC};
use graphlite_core::vector::quantization::QuantizedVector;

#[test]
fn test_large_scale_storage_roundtrip_fidelity() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("production_knowledge.graph");

    let mut interner = StringInterner::new();
    let n_nodes = 50;
    let vector_dim = 64;

    let mut nodes = Vec::with_capacity(n_nodes);
    let mut vectors = Vec::with_capacity(n_nodes);

    let rel_type_id = interner.intern("RELACIONADO_A");

    // 1. Generate multilingual and Unicode nodes + quantized vectors
    for i in 0..n_nodes {
        let name = format!("Entidade_{:03}_🚀_SãoPaulo_東京", i);
        let s_id = interner.intern(&name);

        let node = NodeRecord::new(
            NodeId::new(s_id.as_u32()),
            s_id,
            StringId::new(1),
            StringId::INVALID,
            i as u64 * (8 + vector_dim as u64),
        );
        nodes.push(node);

        let mut data = vec![0i8; vector_dim];
        for (d, val) in data.iter_mut().enumerate() {
            *val = (((i * 7 + d * 3) as i32) % 250 - 125) as i8;
        }
        vectors.push(QuantizedVector {
            data,
            scale: 0.05,
            norm: 1.5, // módulo do vetor
        });
    }

    // 2. Build interconnected CSR graph
    let mut offsets = Vec::with_capacity(n_nodes + 1);
    let mut edges = Vec::new();
    let mut cur_offset = 0u64;
    offsets.push(cur_offset);

    let mut edge_id_counter = 1u32;
    for i in 0..n_nodes {
        // Connect each node to the next two nodes
        if i + 1 < n_nodes {
            edges.push(
                EdgeRecord::new(
                    EdgeId::new(edge_id_counter),
                    nodes[i].id,
                    nodes[i + 1].id,
                    rel_type_id,
                )
                .with_weight(0.9),
            );
            edge_id_counter += 1;
            cur_offset += 1;
        }
        if i + 2 < n_nodes {
            edges.push(
                EdgeRecord::new(
                    EdgeId::new(edge_id_counter),
                    nodes[i].id,
                    nodes[i + 2].id,
                    rel_type_id,
                )
                .with_weight(0.8),
            );
            edge_id_counter += 1;
            cur_offset += 1;
        }
        offsets.push(cur_offset);
    }

    let csr = CsrGraph::new(offsets, edges, n_nodes);

    // 3. Atomically write `.graph` database to disk
    write_database_atomic(
        &db_path, &nodes, &csr, &vectors, &interner, vector_dim, 0, // Cosine metric
    )
    .unwrap();

    assert!(db_path.exists());

    // 4. Open via MmapGraphReader and assert 100% data parity
    let reader = MmapGraphReader::open(&db_path).unwrap();

    assert_eq!(reader.header().node_count as usize, n_nodes);
    assert_eq!(reader.header().edge_count, csr.edge_count() as u32);
    assert_eq!(reader.header().vector_dim as usize, vector_dim);

    // Validate string resolution and node lookups
    for i in 0..n_nodes {
        let expected_name = format!("Entidade_{:03}_🚀_SãoPaulo_東京", i);
        let s_id = nodes[i].name_id;
        assert_eq!(reader.resolve_string(s_id), Some(expected_name.as_str()));

        let node = reader.get_node(nodes[i].id).unwrap();
        assert_eq!(node.id, nodes[i].id);
        assert_eq!(node.name_id, s_id);

        // Validate vector retrieval
        let v = reader.get_vector(i).unwrap();
        assert_eq!(v, vectors[i]);

        // Validate edge traversal parity
        let disk_out_edges = reader.get_out_edges(nodes[i].id);
        let mem_out_edges = csr.out_edges(nodes[i].id);
        assert_eq!(disk_out_edges.len(), mem_out_edges.len());
        for (d_edge, m_edge) in disk_out_edges.iter().zip(mem_out_edges.iter()) {
            assert_eq!(d_edge.target, m_edge.target);
            assert!((d_edge.weight - m_edge.weight).abs() < 1e-4);
        }
    }
}

#[test]
fn test_forced_corruption_and_checksum_verification() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("corrupt_test.graph");

    let mut interner = StringInterner::new();
    let s0 = interner.intern("Nó Seguro");
    let node0 = NodeRecord::new(
        NodeId::new(0),
        s0,
        StringId::new(1),
        StringId::INVALID,
        NO_VECTOR_OFFSET,
    );
    let csr = CsrGraph::new(vec![0, 0], vec![], 1);

    write_database_atomic(&db_path, &[node0], &csr, &[], &interner, 0, 0).unwrap();

    // 1. Initial open must succeed
    assert!(MmapGraphReader::open(&db_path).is_ok());

    // 2. Corrupt a single payload byte in the file
    {
        let mut file = OpenOptions::new().write(true).open(&db_path).unwrap();
        let payload_pos = 100u64; // arbitrary position inside payload
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(payload_pos)).unwrap();
        file.write_all(&[0xFF]).unwrap();
        file.flush().unwrap();
    }

    // 3. Opening must fail with ChecksumMismatch
    let err = MmapGraphReader::open(&db_path).unwrap_err();
    match err {
        GraphLiteError::ChecksumMismatch { .. } => {}
        other => panic!("Expected ChecksumMismatch, got {:?}", other),
    }

    // 4. Corrupt magic bytes
    {
        let mut file = OpenOptions::new().write(true).open(&db_path).unwrap();
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).unwrap();
        file.write_all(b"NOPE").unwrap();
        file.flush().unwrap();
    }

    let err = MmapGraphReader::open(&db_path).unwrap_err();
    match err {
        GraphLiteError::InvalidMagicBytes { expected, found } => {
            assert_eq!(expected, GRAPH_MAGIC);
            assert_eq!(found, *b"NOPE");
        }
        other => panic!("Expected InvalidMagicBytes, got {:?}", other),
    }
}

#[test]
fn test_atomic_overwrite_clean_replacement() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("versioned.graph");

    let mut interner_v1 = StringInterner::new();
    let s_v1 = interner_v1.intern("Versão 1");
    let node_v1 = NodeRecord::new(
        NodeId::new(0),
        s_v1,
        StringId::new(1),
        StringId::INVALID,
        NO_VECTOR_OFFSET,
    );
    let csr_v1 = CsrGraph::new(vec![0, 0], vec![], 1);

    // Write Version 1
    write_database_atomic(&db_path, &[node_v1], &csr_v1, &[], &interner_v1, 0, 0).unwrap();
    let reader_v1 = MmapGraphReader::open(&db_path).unwrap();
    assert_eq!(reader_v1.header().node_count, 1);
    assert_eq!(reader_v1.resolve_string(s_v1), Some("Versão 1"));

    // Overwrite atomically with Version 2
    let mut interner_v2 = StringInterner::new();
    let s_v2 = interner_v2.intern("Versão 2 Totalmente Nova");
    let node_v2 = NodeRecord::new(
        NodeId::new(0),
        s_v2,
        StringId::new(1),
        StringId::INVALID,
        NO_VECTOR_OFFSET,
    );
    let node_v2_extra = NodeRecord::new(
        NodeId::new(1),
        s_v2,
        StringId::new(1),
        StringId::INVALID,
        NO_VECTOR_OFFSET,
    );
    let csr_v2 = CsrGraph::new(vec![0, 0, 0], vec![], 2);

    write_database_atomic(
        &db_path,
        &[node_v2, node_v2_extra],
        &csr_v2,
        &[],
        &interner_v2,
        0,
        0,
    )
    .unwrap();

    let reader_v2 = MmapGraphReader::open(&db_path).unwrap();
    assert_eq!(reader_v2.header().node_count, 2);
    assert_eq!(
        reader_v2.resolve_string(s_v2),
        Some("Versão 2 Totalmente Nova")
    );
}
