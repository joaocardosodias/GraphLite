//! Inverted Index and BM25 Lexical Ranking for Hybrid Keyword + Vector Search.

use std::collections::HashMap;

use crate::id::NodeId;

/// Parameters for BM25 ranking algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (standard default: 1.2).
    pub k1: f32,
    /// Document length normalization parameter (standard default: 0.75).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Inverted index for fast BM25 exact lexical search across knowledge graph entities.
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    /// Mapping from token -> list of (NodeId, TermFrequency)
    inverted_index: HashMap<String, Vec<(NodeId, u32)>>,
    /// Document lengths (number of terms) for each node
    doc_lengths: HashMap<NodeId, usize>,
    /// Total number of indexed terms across all documents
    total_tokens: usize,
    /// BM25 tuning parameters
    params: Bm25Params,
}

impl Bm25Index {
    /// Creates a new empty `Bm25Index`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new `Bm25Index` with custom parameters.
    pub fn with_params(params: Bm25Params) -> Self {
        Self {
            params,
            ..Default::default()
        }
    }

    /// Tokenizes input text into normalized lowercase alphanumeric terms,
    /// with intelligent CamelCase splitting, snake_case splitting, and bilingual code synonym expansion.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // 1. Initial split on whitespace and punctuation (preserving underscore for compound identifiers)
        let raw_parts = text.split(|c: char| !c.is_alphanumeric() && c != '_');

        for part in raw_parts {
            let trimmed = part.trim();
            if trimmed.len() < 2 {
                continue;
            }

            let lower = trimmed.to_lowercase();
            if !tokens.contains(&lower) {
                tokens.push(lower.clone());
            }

            // 1b. If it contains `_`, also extract sub-words
            if trimmed.contains('_') {
                for sub in trimmed.split('_') {
                    let sub_lower = sub.trim().to_lowercase();
                    if sub_lower.len() >= 2 && !tokens.contains(&sub_lower) {
                        tokens.push(sub_lower);
                    }
                }
            }

            // 2. CamelCase / PascalCase splitting (e.g. `connectWiFi` -> `connect`, `wifi`)
            let mut sub_token = String::new();
            let mut chars = trimmed.chars().peekable();
            let mut sub_tokens = Vec::new();

            while let Some(c) = chars.next() {
                if c == '_' {
                    if !sub_token.is_empty() {
                        sub_tokens.push(sub_token.to_lowercase());
                        sub_token = String::new();
                    }
                    continue;
                }
                sub_token.push(c);
                if let Some(&next) = chars.peek() {
                    if c.is_lowercase() && next.is_uppercase() {
                        sub_tokens.push(sub_token.to_lowercase());
                        sub_token = String::new();
                    }
                }
            }
            if !sub_token.is_empty() {
                sub_tokens.push(sub_token.to_lowercase());
            }

            if sub_tokens.len() > 1 {
                for st in sub_tokens {
                    if st.len() >= 2 && !tokens.contains(&st) {
                        tokens.push(st);
                    }
                }
            }

            // 3. Synonym expansion for code & cross-lingual queries (PT <-> EN)
            let synonyms: &[&str] = match lower.as_str() {
                "conectar" | "conexao" | "conexão" | "conectando" => &["connect", "connection"],
                "connect" | "connection" => &["conectar", "conexao", "conexão"],
                "banco" | "db" | "database" => &["banco", "db", "database", "dados"],
                "dados" | "data" => &["dados", "data"],
                "funcao" | "função" => &["function", "fn", "func", "method"],
                "function" | "func" | "fn" => &["funcao", "função"],
                "modelo" | "modelos" => &["model", "struct", "schema"],
                "model" | "models" => &["modelo", "modelos", "struct"],
                "struct" | "structs" => &["struct", "structs", "modelo", "modelos", "estrutura"],
                "classe" | "classes" => &["class", "classes"],
                "class" => &["classe", "classes"],
                "tabela" | "tabelas" => &["table", "tables"],
                "table" | "tables" => &["tabela", "tabelas"],
                "rota" | "rotas" | "endpoint" | "endpoints" => &["route", "endpoint", "api"],
                "cdc" => &["consumidor", "consumo", "defesa"],
                "lgpd" => &["lgpd", "privacidade", "dados", "pessoais"],
                "cf" | "cf88" => &["constituicao", "constituição", "federal"],
                _ => &[],
            };

            for &syn in synonyms {
                let syn_str = syn.to_string();
                if !tokens.contains(&syn_str) {
                    tokens.push(syn_str);
                }
            }
        }

        tokens
    }

    /// Indexes a node with its associated text (name, type, description).
    pub fn index_node(&mut self, node_id: NodeId, text: &str) {
        // Remove prior entry if re-indexing
        self.remove_node(node_id);

        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let doc_len = tokens.len();
        self.doc_lengths.insert(node_id, doc_len);
        self.total_tokens += doc_len;

        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token).or_insert(0) += 1;
        }

        for (token, tf) in tf_map {
            self.inverted_index
                .entry(token)
                .or_default()
                .push((node_id, tf));
        }
    }

    /// Removes a node from the inverted index.
    pub fn remove_node(&mut self, node_id: NodeId) {
        if let Some(old_len) = self.doc_lengths.remove(&node_id) {
            self.total_tokens = self.total_tokens.saturating_sub(old_len);
            for postings in self.inverted_index.values_mut() {
                postings.retain(|(nid, _)| *nid != node_id);
            }
            self.inverted_index
                .retain(|_, postings| !postings.is_empty());
        }
    }

    /// Number of documents (nodes) currently indexed.
    #[inline]
    pub fn doc_count(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Average document length across the indexed collection.
    #[inline]
    pub fn avg_doc_len(&self) -> f32 {
        let n = self.doc_lengths.len();
        if n == 0 {
            0.0
        } else {
            self.total_tokens as f32 / n as f32
        }
    }

    /// Searches the BM25 index with a textual query, returning ranked (NodeId, Score) pairs.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(NodeId, f32)> {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() || self.doc_lengths.is_empty() {
            return Vec::new();
        }

        let n_docs = self.doc_lengths.len() as f32;
        let avg_dl = self.avg_doc_len();
        let k1 = self.params.k1;
        let b = self.params.b;

        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len() as f32;
                // Standard Robertson-Spärck Jones IDF
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(nid, tf) in postings {
                    let doc_len = *self.doc_lengths.get(&nid).unwrap_or(&1) as f32;
                    let numerator = (tf as f32) * (k1 + 1.0);
                    let denominator = (tf as f32) + k1 * (1.0 - b + b * (doc_len / avg_dl));
                    let term_score = idf * (numerator / denominator);

                    *scores.entry(nid).or_insert(0.0) += term_score;
                }
            }
        }

        let mut ranked: Vec<(NodeId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }
}

/// Reciprocal Rank Fusion (RRF) algorithm combining dense vector ranks and BM25 lexical ranks.
///
/// Score formula: `RRF(d) = (1 / (k + rank_vector(d))) + (1 / (k + rank_bm25(d)))`
/// where `k = 60` is the standard smoothing constant.
pub fn reciprocal_rank_fusion(
    vector_ranked: &[NodeId],
    bm25_ranked: &[NodeId],
    k: usize,
) -> Vec<(NodeId, f32)> {
    let mut scores: HashMap<NodeId, f32> = HashMap::new();

    for (rank, &nid) in vector_ranked.iter().enumerate() {
        let rrf_contribution = 1.0 / ((k + rank + 1) as f32);
        *scores.entry(nid).or_insert(0.0) += rrf_contribution;
    }

    for (rank, &nid) in bm25_ranked.iter().enumerate() {
        let rrf_contribution = 1.0 / ((k + rank + 1) as f32);
        *scores.entry(nid).or_insert(0.0) += rrf_contribution;
    }

    let mut fused: Vec<(NodeId, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_tokenization() {
        let tokens = Bm25Index::tokenize("AuthService::validate_token(jwt_token: &str)");
        assert!(tokens.contains(&"authservice".to_string()));
        assert!(tokens.contains(&"validate_token".to_string()));
        assert!(tokens.contains(&"jwt_token".to_string()));
        assert!(tokens.contains(&"str".to_string()));
    }

    #[test]
    fn test_bm25_indexing_and_search() {
        let mut index = Bm25Index::new();

        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);
        let n3 = NodeId::new(3);

        index.index_node(
            n1,
            "AuthService JWT token validation and session management",
        );
        index.index_node(n2, "PaymentGateway Stripe and Pix transaction processing");
        index.index_node(n3, "InventoryManager real-time stock and warehouse catalog");

        assert_eq!(index.doc_count(), 3);

        let results = index.search("JWT validation", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, n1);

        let results_payment = index.search("Stripe Pix", 5);
        assert_eq!(results_payment[0].0, n2);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);
        let n3 = NodeId::new(3);

        let vector_ranked = vec![n1, n2, n3];
        let bm25_ranked = vec![n2, n1];

        let fused = reciprocal_rank_fusion(&vector_ranked, &bm25_ranked, 60);
        assert_eq!(fused.len(), 3);
        // n1 and n2 are in both lists, so they must be scored higher than n3
        assert!(fused[0].0 == n1 || fused[0].0 == n2);
        assert_eq!(fused[2].0, n3);
    }
}
