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

/// Normalizes accented characters and diacritics to ASCII equivalents for robust multi-language search.
#[inline]
pub fn fold_accents(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'ã' | 'â' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'Á' | 'À' | 'Ã' | 'Â' | 'Ä'
            | 'Å' | 'Ā' | 'Ă' | 'Ą' => 'a',
            'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' | 'ě' | 'É' | 'È' | 'Ê' | 'Ë' | 'Ē' | 'Ė'
            | 'Ę' | 'Ě' => 'e',
            'í' | 'ì' | 'î' | 'ï' | 'ī' | 'į' | 'Í' | 'Ì' | 'Î' | 'Ï' | 'Ī' | 'Į' => {
                'i'
            }
            'ó' | 'ò' | 'õ' | 'ô' | 'ö' | 'ø' | 'ō' | 'ő' | 'Ó' | 'Ò' | 'Õ' | 'Ô' | 'Ö' | 'Ø'
            | 'Ō' | 'Ő' => 'o',
            'ú' | 'ù' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'ų' | 'Ú' | 'Ù' | 'Û' | 'Ü' | 'Ū' | 'Ů'
            | 'Ű' | 'Ų' => 'u',
            'ç' | 'ć' | 'č' | 'Ç' | 'Ć' | 'Č' => 'c',
            'ñ' | 'ń' | 'ň' | 'Ñ' | 'Ń' | 'Ň' => 'n',
            'ý' | 'ÿ' | 'Ý' | 'Ÿ' => 'y',
            _ => c,
        })
        .collect()
}

/// Applies lightweight Snowball / RSLP-inspired stemming for Portuguese terms.
pub fn stem_portuguese(w: &str) -> Option<String> {
    if w.len() <= 3 {
        return None;
    }

    let mut s = w.to_string();

    // 1. Plural & Suffix Reductions
    if s.ends_with("oes") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("ao");
        return Some(s);
    }
    if s.ends_with("aes") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("ao");
        return Some(s);
    }
    if s.ends_with("ais") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("al");
        return Some(s);
    }
    if s.ends_with("eis") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("el");
        return Some(s);
    }
    if s.ends_with("ois") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("ol");
        return Some(s);
    }
    if s.ends_with("uis") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push_str("ul");
        return Some(s);
    }
    if s.ends_with("res") && s.len() >= 5 {
        s.truncate(s.len() - 2);
        return Some(s);
    }
    if s.ends_with("zes") && s.len() >= 5 {
        s.truncate(s.len() - 2);
        return Some(s);
    }
    if s.ends_with("nes") && s.len() >= 5 {
        s.truncate(s.len() - 2);
        return Some(s);
    }
    if s.ends_with("mente") && s.len() >= 7 {
        s.truncate(s.len() - 5);
        return Some(s);
    }
    if (s.ends_with("issimo") || s.ends_with("issima")) && s.len() >= 8 {
        s.truncate(s.len() - 6);
        return Some(s);
    }
    if (s.ends_with("issimos") || s.ends_with("issimas")) && s.len() >= 9 {
        s.truncate(s.len() - 7);
        return Some(s);
    }

    // 2. Verb & Participle Reductions
    let verb_suffixes = [
        "ando", "endo", "indo", "aram", "eram", "iram", "avam", "asse", "esse", "isse", "aria",
        "eria", "iria", "ados", "adas", "idos", "idas", "ado", "ada", "ido", "ida", "ara", "era",
        "ira", "ava", "iam", "emos", "imos", "amos", "eis", "ais", "ou", "eu", "iu", "ar", "er",
        "ir", "em", "am",
    ];

    for suffix in verb_suffixes {
        if s.ends_with(suffix) && s.len() >= suffix.len() + 3 {
            s.truncate(s.len() - suffix.len());
            return Some(s);
        }
    }

    // 3. Simple plural 's' at the end of vowels
    if s.ends_with('s') && s.len() >= 4 {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() >= 2 {
            let prev_char = chars[chars.len() - 2];
            if matches!(prev_char, 'a' | 'e' | 'i' | 'o' | 'u') {
                s.truncate(s.len() - 1);
                return Some(s);
            }
        }
    }

    None
}

/// Applies standard English suffix stemming.
pub fn stem_english(w: &str) -> Option<String> {
    if w.len() <= 3 {
        return None;
    }

    let mut s = w.to_string();
    if s.ends_with("ing") && s.len() >= 6 {
        s.truncate(s.len() - 3);
        return Some(s);
    }
    if s.ends_with("tion") && s.len() >= 6 {
        s.truncate(s.len() - 4);
        return Some(s);
    }
    if s.ends_with("tions") && s.len() >= 7 {
        s.truncate(s.len() - 5);
        return Some(s);
    }
    if s.ends_with("ies") && s.len() >= 5 {
        s.truncate(s.len() - 3);
        s.push('y');
        return Some(s);
    }
    if s.ends_with("ed") && s.len() >= 5 {
        s.truncate(s.len() - 2);
        return Some(s);
    }
    if s.ends_with("es") && s.len() >= 5 {
        s.truncate(s.len() - 2);
        return Some(s);
    }
    if s.ends_with('s') && s.len() >= 4 && !s.ends_with("ss") {
        s.truncate(s.len() - 1);
        return Some(s);
    }

    None
}

/// Returns true if a token is a common grammatical stopword.
#[inline]
pub fn is_stopword(w: &str) -> bool {
    matches!(
        w,
        "a" | "ao"
            | "aos"
            | "as"
            | "com"
            | "da"
            | "das"
            | "de"
            | "del"
            | "dele"
            | "deles"
            | "dela"
            | "delas"
            | "deste"
            | "desta"
            | "destes"
            | "destas"
            | "desse"
            | "dessa"
            | "desses"
            | "dessas"
            | "diz"
            | "do"
            | "dos"
            | "e"
            | "ela"
            | "elas"
            | "ele"
            | "eles"
            | "em"
            | "era"
            | "eram"
            | "essa"
            | "essas"
            | "esse"
            | "esses"
            | "esta"
            | "estas"
            | "este"
            | "estes"
            | "eu"
            | "fala"
            | "foi"
            | "foram"
            | "ha"
            | "isso"
            | "isto"
            | "la"
            | "lhe"
            | "lhes"
            | "me"
            | "meu"
            | "meus"
            | "minha"
            | "minhas"
            | "na"
            | "nas"
            | "no"
            | "nos"
            | "nosso"
            | "nossa"
            | "nossos"
            | "nossas"
            | "num"
            | "numa"
            | "nums"
            | "numas"
            | "o"
            | "os"
            | "ou"
            | "para"
            | "pela"
            | "pelas"
            | "pelo"
            | "pelos"
            | "por"
            | "pra"
            | "qual"
            | "quais"
            | "quando"
            | "que"
            | "quem"
            | "sao"
            | "se"
            | "sem"
            | "ser"
            | "seu"
            | "seus"
            | "sob"
            | "sobre"
            | "sua"
            | "suas"
            | "te"
            | "tem"
            | "têm"
            | "teu"
            | "teus"
            | "trata"
            | "tu"
            | "tua"
            | "tuas"
            | "um"
            | "uma"
            | "uns"
            | "umas"
            | "voce"
            | "voces"
            | "vos"
            | "the"
            | "is"
            | "at"
            | "which"
            | "on"
            | "and"
            | "or"
            | "of"
            | "to"
            | "in"
            | "an"
            | "for"
            | "with"
            | "what"
            | "does"
            | "say"
            | "about"
    )
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
    /// with accent folding, stopword removal, number normalization, and code synonym expansion.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let folded = fold_accents(text);

        // Pre-normalize hyphens in alphanumeric codes (e.g. "121-A" -> "121a", "art-121" -> "art121")
        let chars: Vec<char> = folded.chars().collect();
        let mut normalized_text = String::with_capacity(folded.len());
        let len = chars.len();
        for i in 0..len {
            if chars[i] == '-' {
                let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
                let next_is_alpha = i + 1 < len && chars[i + 1].is_ascii_alphabetic();
                let prev_is_alpha = i > 0 && chars[i - 1].is_ascii_alphabetic();
                let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();
                if (prev_is_digit && next_is_alpha) || (prev_is_alpha && next_is_digit) {
                    continue; // Skip hyphen so "121-A" merges to "121a"
                }
            }
            normalized_text.push(chars[i]);
        }

        // 0. Extract normalized alphanumeric article/code tokens (e.g. Art. 121 -> "121", "art121", "artigo121")
        for word in normalized_text.split_whitespace() {
            let lower_w = word.to_lowercase();
            let clean_code: String = lower_w
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect();
            if !clean_code.is_empty()
                && clean_code.len() <= 12
                && clean_code.chars().any(|c| c.is_ascii_digit())
            {
                if !tokens.contains(&clean_code) {
                    tokens.push(clean_code.clone());
                }
                if clean_code.starts_with("artigo") {
                    let stripped = clean_code.trim_start_matches("artigo");
                    if !stripped.is_empty() {
                        if !tokens.contains(&stripped.to_string()) {
                            tokens.push(stripped.to_string());
                        }
                        let art_alias = format!("art{}", stripped);
                        if !tokens.contains(&art_alias) {
                            tokens.push(art_alias);
                        }
                    }
                } else if clean_code.starts_with("art") {
                    let stripped = clean_code.trim_start_matches("art");
                    if !stripped.is_empty() {
                        if !tokens.contains(&stripped.to_string()) {
                            tokens.push(stripped.to_string());
                        }
                        let artigo_alias = format!("artigo{}", stripped);
                        if !tokens.contains(&artigo_alias) {
                            tokens.push(artigo_alias);
                        }
                    }
                } else {
                    let art_alias = format!("art{}", clean_code);
                    if !tokens.contains(&art_alias) {
                        tokens.push(art_alias);
                    }
                    let artigo_alias = format!("artigo{}", clean_code);
                    if !tokens.contains(&artigo_alias) {
                        tokens.push(artigo_alias);
                    }
                }
            }
        }

        // 1. Split on whitespace and non-alphanumeric characters
        let raw_parts = normalized_text.split(|c: char| !c.is_alphanumeric() && c != '_');

        for part in raw_parts {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }

            let lower = trimmed.to_lowercase();
            if is_stopword(&lower) && !lower.chars().any(|c| c.is_ascii_digit()) {
                continue; // Skip non-substantive grammatical stopwords
            }

            if !tokens.contains(&lower) {
                tokens.push(lower.clone());
            }

            // 1a. Portuguese & English Stemming for morphological invariance
            if let Some(stem_pt) = stem_portuguese(&lower) {
                if stem_pt.len() >= 3 && !is_stopword(&stem_pt) && !tokens.contains(&stem_pt) {
                    tokens.push(stem_pt);
                }
            }
            if let Some(stem_en) = stem_english(&lower) {
                if stem_en.len() >= 3 && !is_stopword(&stem_en) && !tokens.contains(&stem_en) {
                    tokens.push(stem_en);
                }
            }

            // 1b. If it contains `_`, also extract sub-words
            if trimmed.contains('_') {
                for sub in trimmed.split('_') {
                    let sub_lower = sub.trim().to_lowercase();
                    if sub_lower.len() >= 2 && !is_stopword(&sub_lower) {
                        if !tokens.contains(&sub_lower) {
                            tokens.push(sub_lower.clone());
                        }
                        if let Some(stem_pt) = stem_portuguese(&sub_lower) {
                            if stem_pt.len() >= 3
                                && !is_stopword(&stem_pt)
                                && !tokens.contains(&stem_pt)
                            {
                                tokens.push(stem_pt);
                            }
                        }
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
                    if st.len() >= 2 && !is_stopword(&st) && !tokens.contains(&st) {
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
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.1);

                // Specific numbers, article numbers and codes receive higher weight
                let is_numeric_or_code = term.chars().any(|c| c.is_ascii_digit());
                let term_multiplier = if is_numeric_or_code { 5.0 } else { 1.0 };

                for &(nid, tf) in postings {
                    let doc_len = *self.doc_lengths.get(&nid).unwrap_or(&1) as f32;
                    let numerator = (tf as f32) * (k1 + 1.0);
                    let denominator = (tf as f32) + k1 * (1.0 - b + b * (doc_len / avg_dl));
                    let term_score = idf * (numerator / denominator) * term_multiplier;

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

    #[test]
    fn test_accent_folding_and_diacritics() {
        assert_eq!(fold_accents("Homicídio"), "Homicidio");
        assert_eq!(fold_accents("Constituição"), "Constituicao");
        assert_eq!(fold_accents("ações penais"), "acoes penais");
        assert_eq!(fold_accents("violência fútil"), "violencia futil");
    }

    #[test]
    fn test_portuguese_stemmer() {
        assert_eq!(stem_portuguese("penais"), Some("penal".to_string()));
        assert_eq!(stem_portuguese("papeis"), Some("papel".to_string()));
        assert_eq!(stem_portuguese("acoes"), Some("acao".to_string()));
        assert_eq!(stem_portuguese("cometido"), Some("comet".to_string()));
        assert_eq!(stem_portuguese("matando"), Some("mat".to_string()));
        assert_eq!(stem_portuguese("qualificada"), Some("qualific".to_string()));
    }

    #[test]
    fn test_cross_morphological_bm25_search() {
        let mut index = Bm25Index::new();
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);

        index.index_node(
            n1,
            "Art. 121: Matar alguem. Pena de reclusao por homicidio qualificado.",
        );
        index.index_node(
            n2,
            "Art. 155: Subtrair coisa alheia movel. Crime de furto simples.",
        );

        // Query with inflected verbs and plurals ("matou", "homicídios qualificados")
        let results = index.search("quem matou em homicídios qualificados", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, n1);
    }
}
