use graphite_core::error::GraphiteError;
use graphite_core::id::{EdgeId, NodeId, StringId};
use graphite_core::interner::StringInterner;
use graphite_core::record::{EdgeRecord, NodeRecord, FLAG_ACTIVE, FLAG_DIRECTED, NO_VECTOR_OFFSET};
use std::collections::{BTreeSet, HashSet};

#[test]
fn test_ids_in_hash_and_btree_sets() {
    let mut hash_set = HashSet::new();
    let mut btree_set = BTreeSet::new();

    for i in 0..100 {
        let node = NodeId::new(i);
        assert!(hash_set.insert(node));
        assert!(btree_set.insert(node));
    }

    // Duplicate insertion must return false
    assert!(!hash_set.insert(NodeId::new(50)));
    assert!(!btree_set.insert(NodeId::new(50)));

    assert_eq!(hash_set.len(), 100);
    assert_eq!(btree_set.len(), 100);

    // Verify ordering in BTreeSet
    let ordered: Vec<u32> = btree_set.into_iter().map(|n| n.as_u32()).collect();
    let expected: Vec<u32> = (0..100).collect();
    assert_eq!(ordered, expected);
}

#[test]
fn test_string_interner_unicode_and_portuguese() {
    let mut interner = StringInterner::new();

    // Testing UTF-8 characters with accents and symbols
    let id_pt1 = interner.intern("São Paulo - Engenharia de Software");
    let id_pt2 = interner.intern("São Paulo - Engenharia de Software");
    let id_pt3 = interner.intern("Configuração de Grafos & IA 🚀");

    assert_eq!(id_pt1, id_pt2);
    assert_ne!(id_pt1, id_pt3);

    assert_eq!(
        interner.resolve(id_pt1),
        Some("São Paulo - Engenharia de Software")
    );
    assert_eq!(
        interner.resolve(id_pt3),
        Some("Configuração de Grafos & IA 🚀")
    );

    // Total byte count must match actual UTF-8 byte lengths
    let expected_bytes =
        "São Paulo - Engenharia de Software".len() + "Configuração de Grafos & IA 🚀".len();
    assert_eq!(interner.total_bytes(), expected_bytes);
    assert_eq!(interner.len(), 2);
}

#[test]
fn test_records_binary_packing_and_alignment() {
    assert_eq!(std::mem::size_of::<NodeRecord>(), 32);
    assert_eq!(std::mem::align_of::<NodeRecord>(), 8);

    assert_eq!(std::mem::size_of::<EdgeRecord>(), 32);
    assert_eq!(std::mem::align_of::<EdgeRecord>(), 8);

    // Verify that contiguous slices of records are perfectly packed (320 bytes for 10 records)
    let nodes: Vec<NodeRecord> = (0..10)
        .map(|i| {
            NodeRecord::new(
                NodeId::new(i),
                StringId::new(i * 2),
                StringId::new(i * 2 + 1),
                StringId::INVALID,
                NO_VECTOR_OFFSET,
            )
        })
        .collect();

    assert_eq!(nodes.len() * NodeRecord::BINARY_SIZE, 320);
}

#[test]
fn test_soft_delete_and_flag_combinations() {
    let mut edge = EdgeRecord::new(
        EdgeId::new(1),
        NodeId::new(10),
        NodeId::new(20),
        StringId::new(99),
    )
    .with_weight(0.92)
    .with_directed(true);

    assert!(edge.is_active());
    assert!(edge.is_directed());
    assert_eq!(edge.flags & FLAG_ACTIVE, FLAG_ACTIVE);
    assert_eq!(edge.flags & FLAG_DIRECTED, FLAG_DIRECTED);

    // Soft delete
    edge.set_active(false);
    assert!(!edge.is_active());
    // Direction flag must remain intact after soft-delete
    assert!(edge.is_directed());

    // Reactivate
    edge.set_active(true);
    assert!(edge.is_active());
    assert!(edge.is_directed());
}

#[test]
fn test_simulated_mini_graph_construction() {
    let mut interner = StringInterner::new();

    // 1. Intern strings
    let str_ana = interner.intern("Ana Silva");
    let str_person = interner.intern("Person");
    let str_titan = interner.intern("Projeto Titan");
    let str_project = interner.intern("Project");
    let str_leads = interner.intern("LEADS");

    // 2. Build nodes
    let node_ana = NodeRecord::new(NodeId::new(1), str_ana, str_person, StringId::INVALID, 0);

    let node_titan = NodeRecord::new(
        NodeId::new(2),
        str_titan,
        str_project,
        StringId::INVALID,
        1536,
    );

    // 3. Build edge
    let edge_lead =
        EdgeRecord::new(EdgeId::new(1), node_ana.id, node_titan.id, str_leads).with_weight(0.98);

    // 4. Verify graph connections and string resolutions
    assert_eq!(interner.resolve(node_ana.name_id), Some("Ana Silva"));
    assert_eq!(interner.resolve(node_titan.name_id), Some("Projeto Titan"));
    assert_eq!(interner.resolve(edge_lead.relation_id), Some("LEADS"));
    assert_eq!(edge_lead.source, node_ana.id);
    assert_eq!(edge_lead.target, node_titan.id);
    assert_eq!(edge_lead.weight, 0.98);
}

#[test]
fn test_error_matching() {
    fn simulate_lookup(id: NodeId) -> Result<(), GraphiteError> {
        if id.as_u32() > 100 {
            Err(GraphiteError::NodeNotFound(id))
        } else {
            Ok(())
        }
    }

    assert!(simulate_lookup(NodeId::new(5)).is_ok());
    let err = simulate_lookup(NodeId::new(500)).unwrap_err();
    match err {
        GraphiteError::NodeNotFound(id) => assert_eq!(id, NodeId::new(500)),
        _ => panic!("Expected NodeNotFound error variant"),
    }
}
