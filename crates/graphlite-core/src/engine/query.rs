use crate::engine::instance::GraphLiteEngine;
use crate::error::{GraphLiteError, Result};
use crate::graph::hybrid_score::ScoredEntity;
use crate::graph::subgraph::extract_subgraph_adjacency;
use crate::id::NodeId;
use crate::prompt::markdown::{format_pruned_subgraph_markdown, MarkdownFormatConfig, MarkdownStyle};
use crate::prompt::pruner::{prune_subgraph_by_budget, PrunedSubgraph};
use crate::prompt::token_counter::TiktokenCounter;

/// Optional overrides and settings for a GraphLite context retrieval query.
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// Number of entry seed nodes retrieved via vector search (default: 5).
    pub top_k_seeds: usize,
    /// Optional plain text query for BM25 lexical search and RRF fusion.
    pub query_text: Option<String>,
    /// Maximum token budget for the returned prompt context (default: from `GraphLiteConfig`).
    pub max_tokens: Option<usize>,
    /// Markdown rendering style (Hierarchical vs SeparatedSections).
    pub markdown_style: MarkdownStyle,
    /// Maximum BFS graph exploration depth in hops.
    pub max_depth: Option<usize>,
    /// Minimum hybrid score threshold for entities to be included.
    pub min_score_threshold: Option<f32>,
    /// Alpha balance between vector score and graph topology score ($0.0 \le \alpha \le 1.0$).
    pub alpha: Option<f32>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            top_k_seeds: 5,
            query_text: None,
            max_tokens: None,
            markdown_style: MarkdownStyle::Hierarchical,
            max_depth: None,
            min_score_threshold: None,
            alpha: None,
        }
    }
}

/// The structured result of an end-to-end GraphLite retrieval query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Formatted, LLM-ready Markdown string strictly within the token budget.
    pub markdown: String,
    /// Exact token count of the generated Markdown context.
    pub token_count: usize,
    /// Number of entities included in the pruned context.
    pub entities_count: usize,
    /// Number of relational edges included in the pruned context.
    pub edges_count: usize,
    /// Ranked entities with detailed vector, graph, depth, and hybrid scores.
    pub scored_entities: Vec<ScoredEntity>,
    /// The pruned subgraph data structure containing the retained nodes and cross-edges.
    pub pruned_subgraph: PrunedSubgraph,
}

impl GraphLiteEngine {
    /// Executes a direct vector search returning the Top-K closest nodes.
    pub fn search_vectors(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        let state = self.state.read();
        state.vectors.search(query_vector, top_k)
    }

    /// Executes a direct BM25 keyword search returning ranked nodes.
    pub fn search_bm25(&self, query_text: &str, top_k: usize) -> Vec<(NodeId, f32)> {
        let state = self.state.read();
        state.bm25.search(query_text, top_k)
    }

    /// End-to-end GraphRAG query: Query Vector + BM25 $\to$ RRF Seeds $\to$ Multi-Hop BFS $\to$ Hybrid Scoring $\to$ Subgraph $\to$ Budget Pruning $\to$ Markdown.
    pub fn retrieve_context(
        &self,
        query_vector: &[f32],
        options: Option<QueryOptions>,
    ) -> Result<QueryResult> {
        let opts = options.unwrap_or_default();
        let state = self.state.read();

        if query_vector.len() != self.config.vector_dim {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.config.vector_dim,
                found: query_vector.len(),
            });
        }

        // 1. Vector Search + BM25 Lexical Hybrid Search via RRF Fusion
        let seed_matches = if let Some(ref text) = opts.query_text {
            let bm25_matches = state.bm25.search(text, opts.top_k_seeds * 2);
            let vector_matches = state.vectors.search(query_vector, opts.top_k_seeds * 2)?;

            let vec_ids: Vec<NodeId> = vector_matches.iter().map(|(id, _)| *id).collect();
            let bm25_ids: Vec<NodeId> = bm25_matches.iter().map(|(id, _)| *id).collect();

            let fused = crate::graph::bm25::reciprocal_rank_fusion(&vec_ids, &bm25_ids, 60);
            let mut top_fused = Vec::new();
            for (id, rrf_score) in fused.into_iter().take(opts.top_k_seeds) {
                let score = vector_matches
                    .iter()
                    .find(|(nid, _)| *nid == id)
                    .map(|(_, s)| *s)
                    .unwrap_or(rrf_score.min(1.0));
                top_fused.push((id, score));
            }
            top_fused
        } else {
            state.vectors.search(query_vector, opts.top_k_seeds)?
        };

        if seed_matches.is_empty() {
            let budget = opts.max_tokens.unwrap_or(self.config.default_max_tokens);
            return Ok(QueryResult {
                markdown: String::new(),
                token_count: 0,
                entities_count: 0,
                edges_count: 0,
                scored_entities: Vec::new(),
                pruned_subgraph: PrunedSubgraph {
                    entities: Vec::new(),
                    edges: Vec::new(),
                    total_tokens: 0,
                    budget,
                },
            });
        }

        // 2. Configure Traversal and Hybrid Parameters
        let mut traversal_config = self.config.traversal_config.clone();
        if let Some(depth) = opts.max_depth {
            traversal_config.max_depth = depth;
        }

        let mut hybrid_config = self.config.hybrid_config.clone();
        if let Some(alpha) = opts.alpha {
            hybrid_config.alpha = alpha.clamp(0.0, 1.0);
        }
        if let Some(threshold) = opts.min_score_threshold {
            hybrid_config.min_score_threshold = threshold;
        }

        // 3. Extract Connected Subgraph with Lateral Cross-Edges
        let connected_subgraph = extract_subgraph_adjacency(
            &state.graph,
            &seed_matches,
            &traversal_config,
            &hybrid_config,
        );

        // 4. Prune Subgraph by Token Budget
        let token_budget = opts.max_tokens.unwrap_or(self.config.default_max_tokens);
        let token_counter = TiktokenCounter::cl100k();
        let pruned_subgraph = prune_subgraph_by_budget(
            &connected_subgraph,
            &state.interner,
            token_budget,
            &token_counter,
        );

        // 5. Format Final Markdown for LLM Prompt
        let format_config = MarkdownFormatConfig {
            header_title: "Contexto Recuperado do Conhecimento".to_string(),
            include_scores: true,
            include_edge_weights: true,
            style: opts.markdown_style,
        };
        let markdown = format_pruned_subgraph_markdown(&pruned_subgraph, &state.interner, &format_config);

        let entities_count = pruned_subgraph.entities.len();
        let edges_count = pruned_subgraph.edges.len();
        let token_count = pruned_subgraph.total_tokens;
        let scored_entities = connected_subgraph.entities;

        Ok(QueryResult {
            markdown,
            token_count,
            entities_count,
            edges_count,
            scored_entities,
            pruned_subgraph,
        })
    }

    /// Retrieves context starting from explicit entity names (Textual Seed Exploration).
    pub fn retrieve_context_by_seed_names(
        &self,
        seed_names: &[&str],
        options: Option<QueryOptions>,
    ) -> Result<QueryResult> {
        let opts = options.unwrap_or_default();
        let state = self.state.read();

        let mut seed_matches = Vec::new();
        for &name in seed_names {
            if let Some(name_id) = state.interner.get_id(name.trim()) {
                if let Some(node) = state.graph.nodes().find(|n| n.name_id == name_id) {
                    seed_matches.push((node.id, 1.0f32));
                }
            }
        }

        let budget = opts.max_tokens.unwrap_or(self.config.default_max_tokens);
        if seed_matches.is_empty() {
            return Ok(QueryResult {
                markdown: String::new(),
                token_count: 0,
                entities_count: 0,
                edges_count: 0,
                scored_entities: Vec::new(),
                pruned_subgraph: PrunedSubgraph {
                    entities: Vec::new(),
                    edges: Vec::new(),
                    total_tokens: 0,
                    budget,
                },
            });
        }

        // Traversal and Hybrid Parameters
        let mut traversal_config = self.config.traversal_config.clone();
        if let Some(depth) = opts.max_depth {
            traversal_config.max_depth = depth;
        }

        let mut hybrid_config = self.config.hybrid_config.clone();
        if let Some(alpha) = opts.alpha {
            hybrid_config.alpha = alpha.clamp(0.0, 1.0);
        }
        if let Some(threshold) = opts.min_score_threshold {
            hybrid_config.min_score_threshold = threshold;
        }

        // Extract Connected Subgraph
        let connected_subgraph = extract_subgraph_adjacency(
            &state.graph,
            &seed_matches,
            &traversal_config,
            &hybrid_config,
        );

        // Budget Pruning
        let token_counter = TiktokenCounter::cl100k();
        let pruned_subgraph = prune_subgraph_by_budget(
            &connected_subgraph,
            &state.interner,
            budget,
            &token_counter,
        );

        // Markdown Formatting
        let format_config = MarkdownFormatConfig {
            header_title: "Contexto Recuperado do Conhecimento".to_string(),
            include_scores: true,
            include_edge_weights: true,
            style: opts.markdown_style,
        };
        let markdown = format_pruned_subgraph_markdown(&pruned_subgraph, &state.interner, &format_config);

        let entities_count = pruned_subgraph.entities.len();
        let edges_count = pruned_subgraph.edges.len();
        let token_count = pruned_subgraph.total_tokens;
        let scored_entities = connected_subgraph.entities;

        Ok(QueryResult {
            markdown,
            token_count,
            entities_count,
            edges_count,
            scored_entities,
            pruned_subgraph,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::GraphLiteConfig;

    #[test]
    fn test_end_to_end_retrieve_context() {
        let config = GraphLiteConfig::new().with_dim(4).with_max_tokens(1000);
        let engine = GraphLiteEngine::in_memory(config).unwrap();

        let v_titan = [1.0, 0.0, 0.0, 0.0];
        let v_ana = [0.9, 0.1, 0.0, 0.0];
        let v_carlos = [0.0, 1.0, 0.0, 0.0];

        let id_titan = engine
            .upsert_node("Projeto Titan", "Projeto", "IA Core", Some(&v_titan))
            .unwrap();
        let id_ana = engine
            .upsert_node("Ana Silva", "Pessoa", "Tech Lead", Some(&v_ana))
            .unwrap();
        let id_carlos = engine
            .upsert_node("Carlos Dev", "Pessoa", "Rust Engineer", Some(&v_carlos))
            .unwrap();

        engine.add_edge(id_titan, id_ana, "TEM_LIDER", 0.95, true).unwrap();
        engine.add_edge(id_titan, id_carlos, "TEM_ENGENHEIRO", 0.85, true).unwrap();
        engine.add_edge(id_ana, id_titan, "LIDERA", 0.95, true).unwrap();
        engine.add_edge(id_carlos, id_titan, "DESENVOLVE", 0.85, true).unwrap();

        // Query with vector matching "Projeto Titan"
        let query_vector = [0.98, 0.02, 0.0, 0.0];
        let result = engine.retrieve_context(&query_vector, None).unwrap();

        assert!(!result.markdown.is_empty());
        assert!(result.markdown.contains("Projeto Titan"));
        assert!(result.markdown.contains("Ana Silva"));
        assert!(result.entities_count >= 2);
        assert!(result.token_count > 0 && result.token_count <= 1000);

        // Query by seed name directly
        let name_result = engine
            .retrieve_context_by_seed_names(&["Projeto Titan"], None)
            .unwrap();

        assert!(name_result.markdown.contains("Projeto Titan"));
        assert!(name_result.markdown.contains("Ana Silva"));
    }
}
