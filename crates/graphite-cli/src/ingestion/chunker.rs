//! Hierarchical Semantic Text Chunker for Knowledge Graph construction.

use std::path::Path;

/// A parsed text chunk ready for knowledge graph node creation.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub title: String,
    pub chunk_type: String,
    pub content: String,
    pub file_path: String,
    pub line_number: usize,
    pub section_hierarchy: Vec<String>,
    pub relations: Vec<(String, String, f32)>, // (target_node_name, relation, weight)
}

/// Configuration options for the semantic chunker.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub target_chars: usize,
    pub overlap_chars: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_chars: 1200, // ~300-350 tokens
            overlap_chars: 150, // ~40 tokens
        }
    }
}

/// Parses a Markdown or plain text document into hierarchical semantic chunks.
pub fn chunk_markdown_document(
    content: &str,
    file_path: &str,
    file_hash: &str,
    config: &ChunkConfig,
) -> Vec<DocumentChunk> {
    let mut chunks = Vec::new();
    let file_basename = Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);

    let doc_root_name = format!("Doc: {}", file_basename);

    // 1. Create root document node
    chunks.push(DocumentChunk {
        chunk_id: doc_root_name.clone(),
        title: file_basename.to_string(),
        chunk_type: "Document".to_string(),
        content: format!(
            "Knowledge document: {} | Hash: {}",
            file_basename, file_hash
        ),
        file_path: file_path.to_string(),
        line_number: 1,
        section_hierarchy: vec![file_basename.to_string()],
        relations: Vec::new(),
    });

    let lines: Vec<&str> = content.lines().collect();

    let mut current_h1: Option<String> = None;
    let mut current_h2: Option<String> = None;
    let mut current_h3: Option<String> = None;
    let mut current_section_title = file_basename.to_string();
    let mut current_section_lines: Vec<(usize, String)> = Vec::new();

    let flush_section = |lines_buf: &[(usize, String)],
                         h1: &Option<String>,
                         h2: &Option<String>,
                         _h3: &Option<String>,
                         sec_title: &str,
                         out: &mut Vec<DocumentChunk>| {
        if lines_buf.is_empty() {
            return;
        }

        let section_start_line = lines_buf.first().map(|(l, _)| *l).unwrap_or(1);
        let full_section_text = lines_buf
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if full_section_text.is_empty() {
            return;
        }

        let section_node_name = format!("{}: {}", file_basename, sec_title);

        // Determine parent in hierarchy
        let parent_node = if let Some(ref parent_h2) = h2 {
            if sec_title != parent_h2 {
                format!("{}: {}", file_basename, parent_h2)
            } else if let Some(ref parent_h1) = h1 {
                if sec_title != parent_h1 {
                    format!("{}: {}", file_basename, parent_h1)
                } else {
                    doc_root_name.clone()
                }
            } else {
                doc_root_name.clone()
            }
        } else if let Some(ref parent_h1) = h1 {
            if sec_title != parent_h1 {
                format!("{}: {}", file_basename, parent_h1)
            } else {
                doc_root_name.clone()
            }
        } else {
            doc_root_name.clone()
        };

        let mut breadcrumbs = vec![file_basename.to_string()];
        if let Some(ref parent_h1) = h1 {
            if parent_h1 != file_basename {
                breadcrumbs.push(parent_h1.clone());
            }
        }
        if let Some(ref parent_h2) = h2 {
            if !breadcrumbs.contains(parent_h2) {
                breadcrumbs.push(parent_h2.clone());
            }
        }
        if !breadcrumbs.contains(&sec_title.to_string()) {
            breadcrumbs.push(sec_title.to_string());
        }
        let breadcrumb_header = breadcrumbs.join(" > ");

        // Create Section Node
        let section_relations = vec![
            (parent_node.clone(), "SECTION_OF".to_string(), 0.95),
            (doc_root_name.clone(), "PART_OF_DOC".to_string(), 0.90),
        ];

        out.push(DocumentChunk {
            chunk_id: section_node_name.clone(),
            title: sec_title.to_string(),
            chunk_type: "Section".to_string(),
            content: format!("[{}]\n{}", breadcrumb_header, full_section_text),
            file_path: file_path.to_string(),
            line_number: section_start_line,
            section_hierarchy: breadcrumbs.clone(),
            relations: section_relations,
        });

        // Split long multi-paragraph sections into granular Chunks
        let paragraphs = split_into_semantic_paragraphs(&full_section_text, config);
        if paragraphs.len() > 1 {
            let mut prev_chunk_id: Option<String> = None;

            for (p_idx, paragraph_text) in paragraphs.into_iter().enumerate() {
                let chunk_node_name =
                    format!("{}: {} (Part {})", file_basename, sec_title, p_idx + 1);

                let mut chunk_relations = vec![
                    (section_node_name.clone(), "CHUNK_OF".to_string(), 0.95),
                    (doc_root_name.clone(), "PART_OF_DOC".to_string(), 0.85),
                ];

                if let Some(prev) = prev_chunk_id {
                    chunk_relations.push((prev, "FOLLOWS".to_string(), 0.90));
                }

                prev_chunk_id = Some(chunk_node_name.clone());

                out.push(DocumentChunk {
                    chunk_id: chunk_node_name,
                    title: format!("{} (Part {})", sec_title, p_idx + 1),
                    chunk_type: "Chunk".to_string(),
                    content: format!("[{}]\n{}", breadcrumb_header, paragraph_text.trim()),
                    file_path: file_path.to_string(),
                    line_number: section_start_line,
                    section_hierarchy: breadcrumbs.clone(),
                    relations: chunk_relations,
                });
            }
        }
    };

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            let raw_title = trimmed.trim_start_matches('#').trim();
            let clean_title = clean_heading_title(raw_title);

            if !clean_title.is_empty() {
                flush_section(
                    &current_section_lines,
                    &current_h1,
                    &current_h2,
                    &current_h3,
                    &current_section_title,
                    &mut chunks,
                );
                current_section_lines.clear();

                match level {
                    1 => {
                        current_h1 = Some(clean_title.clone());
                        current_h2 = None;
                        current_h3 = None;
                        current_section_title = clean_title;
                    }
                    2 => {
                        current_h2 = Some(clean_title.clone());
                        current_h3 = None;
                        current_section_title = clean_title;
                    }
                    3..=6 => {
                        current_h3 = Some(clean_title.clone());
                        current_section_title = clean_title;
                    }
                    _ => {}
                }
                continue;
            }
        }

        current_section_lines.push((idx + 1, line.to_string()));
    }

    flush_section(
        &current_section_lines,
        &current_h1,
        &current_h2,
        &current_h3,
        &current_section_title,
        &mut chunks,
    );

    chunks
}

/// Parses plain text, CSV, or JSON content into structured chunks.
pub fn chunk_plain_document(
    content: &str,
    file_path: &str,
    file_hash: &str,
    config: &ChunkConfig,
) -> Vec<DocumentChunk> {
    let mut chunks = Vec::new();
    let file_basename = Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);

    let doc_root_name = format!("Doc: {}", file_basename);

    // Root node
    chunks.push(DocumentChunk {
        chunk_id: doc_root_name.clone(),
        title: file_basename.to_string(),
        chunk_type: "Document".to_string(),
        content: format!(
            "Knowledge document: {} | Hash: {}",
            file_basename, file_hash
        ),
        file_path: file_path.to_string(),
        line_number: 1,
        section_hierarchy: vec![file_basename.to_string()],
        relations: Vec::new(),
    });

    let paragraphs = split_into_semantic_paragraphs(content, config);
    let mut prev_chunk_id: Option<String> = None;

    for (idx, p_text) in paragraphs.into_iter().enumerate() {
        let chunk_node_name = format!("{}: Chunk {}", file_basename, idx + 1);
        let mut relations = vec![(doc_root_name.clone(), "CHUNK_OF".to_string(), 0.95)];

        if let Some(prev) = prev_chunk_id {
            relations.push((prev, "FOLLOWS".to_string(), 0.85));
        }

        prev_chunk_id = Some(chunk_node_name.clone());

        chunks.push(DocumentChunk {
            chunk_id: chunk_node_name,
            title: format!("{} (Chunk {})", file_basename, idx + 1),
            chunk_type: "Chunk".to_string(),
            content: format!("[{}: Chunk {}]\n{}", file_basename, idx + 1, p_text.trim()),
            file_path: file_path.to_string(),
            line_number: 1,
            section_hierarchy: vec![file_basename.to_string(), format!("Chunk {}", idx + 1)],
            relations,
        });
    }

    chunks
}

/// Splits long text into cohesive paragraphs and articles without arbitrary word slicing.
fn split_into_semantic_paragraphs(text: &str, config: &ChunkConfig) -> Vec<String> {
    if text.len() <= config.target_chars {
        return vec![text.to_string()];
    }

    let mut raw_blocks = Vec::new();
    let mut current_block = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_block.trim().is_empty() {
                raw_blocks.push(current_block.trim().to_string());
                current_block = String::new();
            }
            continue;
        }

        let is_article_start = trimmed.starts_with("Art.")
            || trimmed.starts_with("Artigo ")
            || trimmed.starts_with("Article ")
            || trimmed.starts_with("Section ")
            || trimmed.starts_with("CAPÍTULO")
            || trimmed.starts_with("TÍTULO")
            || trimmed.starts_with("LIVRO")
            || trimmed.starts_with("CHAPTER ");

        if is_article_start && !current_block.trim().is_empty() {
            raw_blocks.push(current_block.trim().to_string());
            current_block = String::new();
        }

        if !current_block.is_empty() {
            current_block.push('\n');
        }
        current_block.push_str(trimmed);
    }

    if !current_block.trim().is_empty() {
        raw_blocks.push(current_block.trim().to_string());
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for block in raw_blocks {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }

        if current_chunk.len() + trimmed.len() > config.target_chars && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(trimmed);
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Cleans markdown heading titles by removing emojis and trailing colons.
fn clean_heading_title(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .filter(|c| {
            c.is_alphanumeric()
                || c.is_whitespace()
                || *c == '-'
                || *c == '_'
                || *c == '.'
                || *c == '('
                || *c == ')'
        })
        .collect();
    let trimmed = cleaned.trim().trim_end_matches(':').trim();
    if trimmed.is_empty() {
        title.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_markdown_hierarchy() {
        let md = r#"
# Manual do Usuário

## Introdução
Este é o guia oficial de integração do Graphite.

## Configuração do RAG
Para configurar o RAG para chatbots, use o comando ingest.
O Graphite particiona documentos em chunks semânticos.
"#;
        let config = ChunkConfig::default();
        let chunks = chunk_markdown_document(md, "manual.md", "abc123hash", &config);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "Document");
        assert!(chunks[0].content.contains("Hash: abc123hash"));
        assert!(chunks.iter().any(|c| c.title == "Introdução"));
        assert!(chunks.iter().any(|c| c.title == "Configuração do RAG"));
    }
}
