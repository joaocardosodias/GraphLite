use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

use graphite::engine::config::GraphiteConfig;
use graphite::engine::entity_resolution::ResolutionConfig;
use graphite::engine::instance::GraphiteEngine;
use graphite::engine::query::QueryOptions;
use graphite::prompt::markdown::MarkdownStyle;
use graphite::vector::distance::Metric;

#[test]
fn test_full_ai_assistant_knowledge_graph_workflow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("ai_assistant.graph");

    let config = GraphiteConfig::new()
        .with_dim(4)
        .with_metric(Metric::Cosine)
        .with_max_tokens(1000)
        .with_alpha(0.65)
        .with_max_depth(2)
        .with_auto_flush(true);

    // 1. Phase 1: Build the knowledge graph
    {
        let db = GraphiteEngine::open_or_create(&db_path, config.clone()).unwrap();

        // Vectors representing semantic clusters
        let v_ai_titan = [1.0, 0.0, 0.0, 0.0]; // AI Cluster
        let v_ana = [0.95, 0.05, 0.0, 0.0]; // AI Cluster
        let v_carlos = [0.9, 0.1, 0.0, 0.0]; // AI Cluster
        let v_graphite = [0.85, 0.15, 0.0, 0.0]; // AI Cluster

        let v_ecommerce = [0.0, 0.0, 1.0, 0.0]; // E-Commerce Cluster (Unrelated)
        let v_mariana = [0.0, 0.0, 0.95, 0.05]; // E-Commerce Cluster
        let v_postgres = [0.0, 0.0, 0.9, 0.1]; // E-Commerce Cluster

        // Insert AI project cluster
        let id_titan = db
            .upsert_node(
                "Projeto Titan",
                "Projeto",
                "IA Generativa e RAG",
                Some(&v_ai_titan),
            )
            .unwrap();
        let id_ana = db
            .upsert_node("Ana Silva", "Pessoa", "Tech Lead de IA", Some(&v_ana))
            .unwrap();
        let id_carlos = db
            .upsert_node("Carlos Dev", "Pessoa", "Engenheiro Rust", Some(&v_carlos))
            .unwrap();
        let id_graphite = db
            .upsert_node(
                "Graphite Engine",
                "Tecnologia",
                "Banco de Grafos Embutido",
                Some(&v_graphite),
            )
            .unwrap();

        // Insert E-Commerce project cluster
        let id_ecom = db
            .upsert_node(
                "Loja Online",
                "Projeto",
                "Plataforma E-Commerce",
                Some(&v_ecommerce),
            )
            .unwrap();
        let id_mariana = db
            .upsert_node(
                "Mariana PM",
                "Pessoa",
                "Gerente de Produto",
                Some(&v_mariana),
            )
            .unwrap();
        let id_postgres = db
            .upsert_node(
                "PostgreSQL",
                "Tecnologia",
                "Banco Relacional",
                Some(&v_postgres),
            )
            .unwrap();

        // Connect relationships
        db.add_edge(id_titan, id_ana, "LIDERADO_POR", 0.95, true)
            .unwrap();
        db.add_edge(id_titan, id_carlos, "DESENVOLVIDO_POR", 0.90, true)
            .unwrap();
        db.add_edge(id_titan, id_graphite, "UTILIZA", 0.99, true)
            .unwrap();
        db.add_edge(id_ana, id_carlos, "COORDENA", 0.85, false)
            .unwrap();

        db.add_edge(id_ecom, id_mariana, "GERENCIADO_POR", 0.90, true)
            .unwrap();
        db.add_edge(id_ecom, id_postgres, "UTILIZA", 0.95, true)
            .unwrap();

        assert_eq!(db.node_count(), 7);
        assert_eq!(db.edge_count(), 6);
    }

    // 2. Phase 2: Reopen from disk via Zero-Copy Mmap and execute query
    {
        let db = GraphiteEngine::open_or_create(&db_path, config).unwrap();
        assert_eq!(db.node_count(), 7);
        assert_eq!(db.edge_count(), 6);

        // Query vector targeting AI cluster: "Quem trabalha em IA e quais ferramentas usam?"
        let query_ai = [0.98, 0.02, 0.0, 0.0];

        let result = db
            .retrieve_context(
                &query_ai,
                Some(QueryOptions {
                    top_k_seeds: 2,
                    query_text: Some("Projeto Titan Ana".to_string()),
                    max_tokens: Some(500),
                    markdown_style: MarkdownStyle::Hierarchical,
                    max_depth: Some(2),
                    min_score_threshold: Some(0.1),
                    alpha: Some(0.6),
                    relative_drop_off: None,
                    redundancy_threshold: None,
                    type_filter: None,
                }),
            )
            .unwrap();

        // Assert Markdown contains the relevant AI entities
        assert!(result.markdown.contains("Projeto Titan"));
        assert!(result.markdown.contains("Ana Silva"));
        assert!(result.markdown.contains("Carlos Dev"));
        assert!(result.markdown.contains("Graphite Engine"));

        // Assert unrelated E-Commerce cluster was pruned out
        assert!(!result.markdown.contains("Loja Online"));
        assert!(!result.markdown.contains("PostgreSQL"));

        // Assert token budget was strictly respected
        assert!(result.token_count > 0 && result.token_count <= 500);
        assert!(result.entities_count >= 3);
    }
}

#[test]
fn test_multithreaded_concurrent_reads_and_writes() {
    let config = GraphiteConfig::new().with_dim(4);
    let engine = Arc::new(GraphiteEngine::in_memory(config).unwrap());

    // Populate initial entities
    for i in 0..10 {
        let name = format!("Entidade_Base_{}", i);
        let v = [(i as f32) * 0.1, 0.2, 0.3, 0.4];
        engine
            .upsert_node(&name, "Base", "Descricao", Some(&v))
            .unwrap();
    }

    let mut handles = Vec::new();

    // Spawn 4 Writer threads
    for t in 0..4 {
        let engine_clone = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..15 {
                let name = format!("Nó_Thread_{}_{}", t, i);
                let v = [0.1 * (t as f32), 0.1 * (i as f32), 0.5, 0.5];
                let _ = engine_clone.upsert_node(&name, "Concorrente", "Mutacao", Some(&v));
            }
        }));
    }

    // Spawn 4 Reader threads
    for _ in 0..4 {
        let engine_clone = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let q = [0.5, 0.5, 0.5, 0.5];
            for _ in 0..20 {
                let res = engine_clone.retrieve_context(&q, None);
                assert!(res.is_ok());
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify engine state remains perfectly valid and accessible
    assert!(engine.node_count() >= 10);
    assert_eq!(engine.vector_count(), engine.node_count());
}

#[test]
fn test_entity_resolution_with_synonyms_and_merging() {
    let config = GraphiteConfig::new().with_dim(4);
    let db = GraphiteEngine::in_memory(config).unwrap();

    let v1 = [1.0, 0.0, 0.0, 0.0];
    let v2 = [0.98, 0.02, 0.0, 0.0]; // Cosine similarity > 0.99 with v1

    let res_config = ResolutionConfig {
        similarity_threshold: 0.90,
        require_matching_type: true,
        merge_descriptions: true,
    };

    // 1. Ingest "Inteligência Artificial Generativa"
    let r1 = db
        .upsert_node_resolved(
            "Inteligência Artificial Generativa",
            "Conceito",
            "Modelos capazes de gerar conteúdo",
            &v1,
            Some(res_config),
        )
        .unwrap();

    assert!(!r1.is_merged);
    assert_eq!(db.node_count(), 1);

    // 2. Ingest synonym "GenAI / IA Generativa"
    let r2 = db
        .upsert_node_resolved(
            "GenAI / IA Generativa",
            "Conceito",
            "Tecnologias como LLMs e difusão",
            &v2,
            Some(res_config),
        )
        .unwrap();

    // Verify it was merged into the existing node
    assert!(r2.is_merged);
    assert_eq!(r2.node_id, r1.node_id);
    assert_eq!(db.node_count(), 1);

    // Verify merged description contains both inputs
    let node = db
        .get_node_by_name("Inteligência Artificial Generativa")
        .unwrap();
    let desc = db.resolve_string(node.description_id).unwrap();
    assert!(desc.contains("gerar conteúdo"));
    assert!(desc.contains("LLMs e difusão"));
}
