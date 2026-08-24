use graphite_core::graph::hybrid_score::ScoredEntity;
use graphite_core::graph::subgraph::ConnectedSubgraph;
use graphite_core::id::{EdgeId, NodeId};
use graphite_core::interner::StringInterner;
use graphite_core::prompt::{
    format_pruned_subgraph_markdown, format_subgraph_json, format_subgraph_triples,
    prune_subgraph_by_budget, MarkdownFormatConfig, MarkdownStyle, TiktokenCounter, TokenCounter,
};
use graphite_core::record::EdgeRecord;

/// Helper to generate a realistic test graph with $N$ ranked entities and interconnected edges.
fn create_mock_subgraph(n_entities: usize) -> (ConnectedSubgraph, StringInterner) {
    let mut interner = StringInterner::new();
    let mut entities = Vec::with_capacity(n_entities);
    let mut edges = Vec::new();

    let rel_conecta = interner.intern("CONECTADO_A");

    for i in 0..n_entities {
        let name = format!("Entidade_{:03}", i);
        let s_id = interner.intern(&name);
        let score = 1.0 - (i as f32) * (0.8 / (n_entities as f32));

        entities.push(ScoredEntity {
            node_id: NodeId::new(s_id.as_u32()),
            final_score: score,
            vector_score: score * 0.9,
            graph_score: score * 0.8,
            depth: i % 3,
            path_edge: None,
            node_record: None,
        });

        // Add some connecting edges
        if i > 0 && i % 2 == 0 {
            let edge = EdgeRecord::new(
                EdgeId::new(i as u32),
                entities[i - 1].node_id,
                entities[i].node_id,
                rel_conecta,
            )
            .with_weight(0.9);
            edges.push(edge);
        }
    }

    let seed_ids = vec![entities[0].node_id];
    let subgraph = ConnectedSubgraph {
        entities,
        edges,
        seed_ids,
    };

    (subgraph, interner)
}

#[test]
fn test_strict_token_budget_compliance_across_scales() {
    let (subgraph, interner) = create_mock_subgraph(40);
    let counter = TiktokenCounter::cl100k();
    let config = MarkdownFormatConfig::default();

    // Test a wide spectrum of token budgets
    let test_budgets = [15, 25, 50, 100, 200, 350, 500, 800, 1500, 3000];

    for &budget in &test_budgets {
        let pruned = prune_subgraph_by_budget(&subgraph, &interner, budget, &counter);

        // 1. Assert strict budget compliance
        assert!(
            pruned.total_tokens <= budget,
            "Exceeded budget! Total: {}, Budget: {}",
            pruned.total_tokens,
            budget
        );

        // 2. Generate actual Markdown and verify that measured tokens never exceed budget
        let markdown = format_pruned_subgraph_markdown(&pruned, &interner, &config);
        let actual_md_tokens = counter.count_tokens(&markdown);

        assert!(
            actual_md_tokens <= budget,
            "Rendered Markdown exceeded budget! Actual: {}, Budget: {}",
            actual_md_tokens,
            budget
        );
    }
}

#[test]
fn test_monotonicity_of_token_budget_expansion() {
    let (subgraph, interner) = create_mock_subgraph(30);
    let counter = TiktokenCounter::cl100k();

    let mut prev_entity_count = 0;
    let mut prev_tokens = 0;

    for budget in [30, 60, 120, 250, 500, 1000] {
        let pruned = prune_subgraph_by_budget(&subgraph, &interner, budget, &counter);

        // Increasing budget must monotonically increase or maintain entity count
        assert!(
            pruned.entity_count() >= prev_entity_count,
            "Entity count dropped from {} to {} with larger budget {}",
            prev_entity_count,
            pruned.entity_count(),
            budget
        );

        // Increasing budget must monotonically increase or maintain token count
        assert!(
            pruned.total_tokens >= prev_tokens,
            "Token count dropped from {} to {} with larger budget {}",
            prev_tokens,
            pruned.total_tokens,
            budget
        );

        prev_entity_count = pruned.entity_count();
        prev_tokens = pruned.total_tokens;
    }
}

#[test]
fn test_json_and_triples_formatting_fidelity() {
    let (subgraph, interner) = create_mock_subgraph(10);
    let counter = TiktokenCounter::cl100k();

    let pruned = prune_subgraph_by_budget(&subgraph, &interner, 500, &counter);

    // JSON Formatting
    let json_str = format_subgraph_json(&pruned, &interner).unwrap();
    assert!(json_str.contains("\"entities\":"));
    assert!(json_str.contains("\"relations\":"));

    // Raw Triples Formatting
    let triples = format_subgraph_triples(&pruned, &interner);
    assert!(!triples.is_empty());
    for t in &triples {
        assert!(t.starts_with('('));
        assert!(t.ends_with(')'));
        assert!(t.contains("CONECTADO_A"));
    }
}

#[test]
fn test_markdown_styling_variations() {
    let (subgraph, interner) = create_mock_subgraph(5);
    let counter = TiktokenCounter::cl100k();
    let pruned = prune_subgraph_by_budget(&subgraph, &interner, 300, &counter);

    // Test Hierarchical Style
    let hier_config = MarkdownFormatConfig {
        style: MarkdownStyle::Hierarchical,
        ..Default::default()
    };
    let hier_md = format_pruned_subgraph_markdown(&pruned, &interner, &hier_config);
    assert!(hier_md.contains("# Retrieved Knowledge Context:"));

    // Test Separated Sections Style
    let sep_config = MarkdownFormatConfig {
        style: MarkdownStyle::SeparatedSections,
        header_title: "Knowledge Base".to_string(),
        include_scores: false,
        include_edge_weights: false,
    };
    let sep_md = format_pruned_subgraph_markdown(&pruned, &interner, &sep_config);
    assert!(sep_md.contains("# Knowledge Base:"));
    assert!(sep_md.contains("## Entities:"));
}
