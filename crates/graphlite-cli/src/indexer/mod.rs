//! Codebase AST & Symbol Indexer for Rust, Python, TypeScript, Go, and Markdown.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// A parsed code symbol ready for insertion into the GraphLite knowledge graph.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ExtractedSymbol {
    pub name: String,
    pub symbol_type: String,
    pub description: String,
    pub file_path: String,
    pub line_number: usize,
    pub parent_symbol: Option<String>,
    pub body: Option<String>,
    pub relations: Vec<(String, String, f32)>, // (target_name, relation_label, weight)
}

/// Recursively scans directory collecting relevant source code files.
pub fn scan_directory<P: AsRef<Path>>(
    root: P,
    allowed_exts: &[&str],
    max_files: usize,
) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs_to_visit = vec![root.as_ref().to_path_buf()];

    while let Some(current_dir) = dirs_to_visit.pop() {
        if let Ok(entries) = fs::read_dir(&current_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                // Skip hidden folders, version control, build caches, and node_modules
                if file_name.starts_with('.')
                    || file_name == "target"
                    || file_name == "node_modules"
                    || file_name == "venv"
                    || file_name == ".venv"
                    || file_name == "dist"
                    || file_name == "build"
                {
                    continue;
                }

                if path.is_dir() {
                    dirs_to_visit.push(path);
                } else if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if allowed_exts.contains(&ext) {
                            files.push(path);
                            if files.len() >= max_files {
                                return files;
                            }
                        }
                    }
                }
            }
        }
    }

    files
}

/// Parses source code file into structured code symbols.
pub fn parse_file(path: &Path) -> anyhow::Result<Vec<ExtractedSymbol>> {
    let content = fs::read_to_string(path)?;
    let relative_path = path.to_string_lossy().to_string();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let symbols = match ext {
        "rs" => parse_rust_file(&content, &relative_path),
        "py" => parse_python_file(&content, &relative_path),
        "ts" | "tsx" | "js" | "jsx" => parse_typescript_file(&content, &relative_path),
        "go" => parse_go_file(&content, &relative_path),
        "md" | "markdown" => parse_markdown_file(&content, &relative_path),
        _ => Vec::new(),
    };

    Ok(symbols)
}

/// Extracts structs, enums, traits, functions, and impl blocks from Rust source code.
fn parse_rust_file(content: &str, file_path: &str) -> Vec<ExtractedSymbol> {
    let mut symbols = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut current_impl: Option<String> = None;
    let mut impl_brace_depth: usize = 0;
    let mut current_docs: Vec<String> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
            current_docs.push(trimmed.trim_start_matches('/').trim().to_string());
            continue;
        }

        // Track `impl Foo` or `impl Trait for Foo`
        if trimmed.starts_with("impl ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.contains(&"for") {
                if let Some(pos) = parts.iter().position(|&x| x == "for") {
                    if let Some(target) = parts.get(pos + 1) {
                        let target_clean = target.trim_end_matches('{').trim();
                        let trait_clean = parts.get(1).unwrap_or(&"").trim();
                        current_impl = Some(target_clean.to_string());

                        symbols.push(ExtractedSymbol {
                            name: format!("{}::{}", target_clean, trait_clean),
                            symbol_type: "TraitImplementation".to_string(),
                            description: format!(
                                "Implementation of trait {} for struct {} ({}:{})",
                                trait_clean,
                                target_clean,
                                file_path,
                                idx + 1
                            ),
                            file_path: file_path.to_string(),
                            line_number: idx + 1,
                            parent_symbol: Some(target_clean.to_string()),
                            body: None,
                            relations: vec![
                                (target_clean.to_string(), "IMPLEMENTS".to_string(), 0.95),
                                (trait_clean.to_string(), "OF_TRAIT".to_string(), 0.90),
                            ],
                        });
                    }
                }
            } else if let Some(struct_name) = parts.get(1) {
                let clean = struct_name.trim_end_matches('{').trim();
                current_impl = Some(clean.to_string());
            }
        }

        if current_impl.is_some() {
            let opens = line.matches('{').count();
            let closes = line.matches('}').count();
            impl_brace_depth += opens;
            impl_brace_depth = impl_brace_depth.saturating_sub(closes);
            if impl_brace_depth == 0 {
                current_impl = None;
            }
        }

        // Match `pub struct Foo` or `struct Foo`
        if (trimmed.starts_with("pub struct ") || trimmed.starts_with("struct "))
            && !trimmed.contains(';')
        {
            let name = extract_keyword_name(trimmed, "struct");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Struct".to_string(),
                    description: format!(
                        "Rust Struct '{}' in {}:{}. {}",
                        name,
                        file_path,
                        idx + 1,
                        doc_summary
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
                current_docs.clear();
            }
        }

        // Match `pub enum Foo` or `enum Foo`
        if (trimmed.starts_with("pub enum ") || trimmed.starts_with("enum "))
            && !trimmed.contains(';')
        {
            let name = extract_keyword_name(trimmed, "enum");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Enum".to_string(),
                    description: format!(
                        "Rust Enum '{}' in {}:{}. {}",
                        name,
                        file_path,
                        idx + 1,
                        doc_summary
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
                current_docs.clear();
            }
        }

        // Match `pub trait Foo` or `trait Foo`
        if (trimmed.starts_with("pub trait ") || trimmed.starts_with("trait "))
            && !trimmed.contains(';')
        {
            let name = extract_keyword_name(trimmed, "trait");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Trait".to_string(),
                    description: format!(
                        "Rust Trait '{}' in {}:{}. {}",
                        name,
                        file_path,
                        idx + 1,
                        doc_summary
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
                current_docs.clear();
            }
        }

        // Match `pub fn foo(` or `fn foo(` or `pub async fn foo(`
        if (trimmed.contains("fn ") && trimmed.contains('(')) && !trimmed.starts_with("//") {
            let name = extract_function_name(trimmed);
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
                let body = extract_block_body(&lines, idx);
                let mut relations = Vec::new();
                if let Some(ref parent) = current_impl {
                    relations.push((parent.clone(), "METHOD_OF".to_string(), 0.95));
                }

                symbols.push(ExtractedSymbol {
                    name: if let Some(ref parent) = current_impl {
                        format!("{}::{}", parent, name)
                    } else {
                        name.clone()
                    },
                    symbol_type: if current_impl.is_some() {
                        "Method".to_string()
                    } else {
                        "Function".to_string()
                    },
                    description: format!(
                        "Rust Function '{}' in {}:{}. Signature: `{}`. {}",
                        name,
                        file_path,
                        idx + 1,
                        trimmed.trim_end_matches('{').trim(),
                        doc_summary
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: current_impl.clone(),
                    body: Some(body),
                    relations,
                });
                current_docs.clear();
            }
        }

        // Match Web Endpoints: `#[get("...")]`, `#[post("...")]`, `.route("/...", ...)`
        if trimmed.starts_with("#[get(")
            || trimmed.starts_with("#[post(")
            || trimmed.starts_with("#[delete(")
            || trimmed.starts_with("#[put(")
        {
            let method = if trimmed.contains("get") {
                "GET"
            } else if trimmed.contains("post") {
                "POST"
            } else if trimmed.contains("delete") {
                "DELETE"
            } else {
                "PUT"
            };
            let endpoint_cleaned = trimmed
                .trim_start_matches('#')
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim();
            symbols.push(ExtractedSymbol {
                name: format!("Endpoint {}", endpoint_cleaned),
                symbol_type: "Endpoint".to_string(),
                description: format!(
                    "HTTP Endpoint {} defined in {}:{}",
                    method,
                    file_path,
                    idx + 1
                ),
                file_path: file_path.to_string(),
                line_number: idx + 1,
                parent_symbol: None,
                body: None,
                relations: Vec::new(),
            });
        }
    }

    symbols
}

/// Extracts Python classes, methods, and functions.
fn parse_python_file(content: &str, file_path: &str) -> Vec<ExtractedSymbol> {
    let mut symbols = Vec::new();
    let mut current_class: Option<(String, usize)> = None;
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let indent = line.len() - line.trim_start().len();

        if let Some((_, c_indent)) = current_class {
            if indent <= c_indent && !trimmed.starts_with("class ") {
                current_class = None;
            }
        }

        if trimmed.starts_with("class ") && trimmed.contains(':') {
            let class_name = trimmed
                .trim_start_matches("class ")
                .split(['(', ':'])
                .next()
                .unwrap_or("")
                .trim();
            if !class_name.is_empty() {
                current_class = Some((class_name.to_string(), indent));
                symbols.push(ExtractedSymbol {
                    name: class_name.to_string(),
                    symbol_type: "Class".to_string(),
                    description: format!(
                        "Python Class '{}' in {}:{}",
                        class_name,
                        file_path,
                        idx + 1
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: None,
                    relations: Vec::new(),
                });
            }
        }

        if trimmed.starts_with("def ") && trimmed.contains('(') {
            let func_name = trimmed
                .trim_start_matches("def ")
                .split('(')
                .next()
                .unwrap_or("")
                .trim();
            if !func_name.is_empty() {
                let is_method = if let Some((_, c_indent)) = &current_class {
                    indent > *c_indent
                } else {
                    false
                };
                let parent = if is_method {
                    current_class.as_ref().map(|(c, _)| c.clone())
                } else {
                    None
                };
                let mut relations = Vec::new();
                if let Some(ref p) = parent {
                    relations.push((p.clone(), "METHOD_OF".to_string(), 0.95));
                }

                let body = extract_python_body(&lines, idx, indent);

                symbols.push(ExtractedSymbol {
                    name: if let Some(ref p) = parent {
                        format!("{}.{}", p, func_name)
                    } else {
                        func_name.to_string()
                    },
                    symbol_type: if is_method {
                        "Method".to_string()
                    } else {
                        "Function".to_string()
                    },
                    description: format!(
                        "Python Function '{}' in {}:{}",
                        func_name,
                        file_path,
                        idx + 1
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: parent,
                    body: Some(body),
                    relations,
                });
            }
        }
    }

    symbols
}

/// Extracts TypeScript/JavaScript interfaces, types, classes, and exported functions.
fn parse_typescript_file(content: &str, file_path: &str) -> Vec<ExtractedSymbol> {
    let mut symbols = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if (trimmed.starts_with("interface ") || trimmed.starts_with("export interface "))
            && trimmed.contains('{')
        {
            let name = trimmed
                .replace("export ", "")
                .replace("interface ", "")
                .split(['<', '{', ' '])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() {
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Interface".to_string(),
                    description: format!(
                        "TypeScript Interface '{}' in {}:{}",
                        name,
                        file_path,
                        idx + 1
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
            }
        }

        if (trimmed.starts_with("type ") || trimmed.starts_with("export type "))
            && trimmed.contains('=')
        {
            let name = trimmed
                .replace("export ", "")
                .replace("type ", "")
                .split(['<', '=', ' '])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() {
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "TypeAlias".to_string(),
                    description: format!("TypeScript Type '{}' in {}:{}", name, file_path, idx + 1),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: None,
                    relations: Vec::new(),
                });
            }
        }

        if (trimmed.starts_with("class ") || trimmed.starts_with("export class "))
            && trimmed.contains('{')
        {
            let name = trimmed
                .replace("export ", "")
                .replace("class ", "")
                .split(['<', '{', ' '])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() {
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Class".to_string(),
                    description: format!(
                        "TypeScript Class '{}' in {}:{}",
                        name,
                        file_path,
                        idx + 1
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
            }
        }

        if (trimmed.starts_with("function ")
            || trimmed.starts_with("export function ")
            || trimmed.starts_with("export const "))
            && trimmed.contains('(')
        {
            let name = trimmed
                .replace("export ", "")
                .replace("async ", "")
                .replace("function ", "")
                .replace("const ", "")
                .split(['(', ':', '=', '<'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();

            if !name.is_empty() && name != "default" {
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.clone(),
                    symbol_type: "Function".to_string(),
                    description: format!(
                        "TypeScript Function '{}' in {}:{}",
                        name,
                        file_path,
                        idx + 1
                    ),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
            }
        }
    }

    symbols
}

/// Extracts Go structs, interfaces, and functions/methods.
fn parse_go_file(content: &str, file_path: &str) -> Vec<ExtractedSymbol> {
    let mut symbols = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("type ") && trimmed.contains("struct") {
            let name = trimmed
                .trim_start_matches("type ")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.to_string(),
                    symbol_type: "Struct".to_string(),
                    description: format!("Go Struct '{}' in {}:{}", name, file_path, idx + 1),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
            }
        }

        if trimmed.starts_with("type ") && trimmed.contains("interface") {
            let name = trimmed
                .trim_start_matches("type ")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                let body = extract_block_body(&lines, idx);
                symbols.push(ExtractedSymbol {
                    name: name.to_string(),
                    symbol_type: "Interface".to_string(),
                    description: format!("Go Interface '{}' in {}:{}", name, file_path, idx + 1),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
                    body: Some(body),
                    relations: Vec::new(),
                });
            }
        }

        if trimmed.starts_with("func ") && trimmed.contains('(') {
            let func_part = trimmed.trim_start_matches("func ");
            let is_receiver = func_part.starts_with('(');

            if is_receiver {
                let receiver_type = func_part
                    .split(')')
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .trim_start_matches('*');
                let method_name = func_part
                    .split(')')
                    .nth(1)
                    .unwrap_or("")
                    .split('(')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !method_name.is_empty() {
                    let body = extract_block_body(&lines, idx);
                    symbols.push(ExtractedSymbol {
                        name: format!("{}.{}", receiver_type, method_name),
                        symbol_type: "Method".to_string(),
                        description: format!(
                            "Go Method '{}.{}' in {}:{}",
                            receiver_type,
                            method_name,
                            file_path,
                            idx + 1
                        ),
                        file_path: file_path.to_string(),
                        line_number: idx + 1,
                        parent_symbol: Some(receiver_type.to_string()),
                        body: Some(body),
                        relations: vec![(receiver_type.to_string(), "METHOD_OF".to_string(), 0.95)],
                    });
                }
            } else {
                let func_name = func_part.split('(').next().unwrap_or("").trim();
                if !func_name.is_empty() {
                    let body = extract_block_body(&lines, idx);
                    symbols.push(ExtractedSymbol {
                        name: func_name.to_string(),
                        symbol_type: "Function".to_string(),
                        description: format!(
                            "Go Function '{}' in {}:{}",
                            func_name,
                            file_path,
                            idx + 1
                        ),
                        file_path: file_path.to_string(),
                        line_number: idx + 1,
                        parent_symbol: None,
                        body: Some(body),
                        relations: Vec::new(),
                    });
                }
            }
        }
    }

    symbols
}

/// Extracts documentation sections, headings, and overview from Markdown files.
fn parse_markdown_file(content: &str, file_path: &str) -> Vec<ExtractedSymbol> {
    let mut symbols = Vec::new();
    let file_basename = Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);

    let doc_root_name = format!("Doc: {}", file_basename);
    let mut current_h2: Option<String> = None;
    let mut current_section_name = doc_root_name.clone();
    let mut current_section_lines: Vec<String> = Vec::new();
    let mut current_section_start = 1;
    let mut current_section_level = 1;

    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            let title = trimmed.trim_start_matches('#').trim();

            if !title.is_empty() {
                if !current_section_lines.is_empty() {
                    let desc = current_section_lines.join("\n").trim().to_string();
                    if !desc.is_empty() {
                        let is_root = current_section_name == doc_root_name;
                        let mut relations = Vec::new();
                        if !is_root {
                            if current_section_level == 3 && current_h2.is_some() {
                                relations.push((
                                    current_h2.clone().unwrap(),
                                    "SUBSECTION_OF".to_string(),
                                    0.90,
                                ));
                            } else {
                                relations.push((
                                    doc_root_name.clone(),
                                    "SECTION_OF".to_string(),
                                    0.95,
                                ));
                            }
                        }

                        symbols.push(ExtractedSymbol {
                            name: current_section_name.clone(),
                            symbol_type: if is_root {
                                "Document".to_string()
                            } else {
                                "DocumentationSection".to_string()
                            },
                            description: format!(
                                "Documentation in {}:{}. Content:\n{}",
                                file_path, current_section_start, desc
                            ),
                            file_path: file_path.to_string(),
                            line_number: current_section_start,
                            parent_symbol: if is_root {
                                None
                            } else {
                                Some(doc_root_name.clone())
                            },
                            body: None,
                            relations,
                        });
                    }
                    current_section_lines.clear();
                }

                current_section_start = idx + 1;
                current_section_level = level;

                if level == 1 {
                    current_section_name = format!("{}: {}", file_basename, title);
                } else if level == 2 {
                    let h2_name = format!("{}: {}", file_basename, title);
                    current_h2 = Some(h2_name.clone());
                    current_section_name = h2_name;
                } else {
                    current_section_name = format!("{}: {}", file_basename, title);
                }
            }
        } else if !trimmed.is_empty() {
            current_section_lines.push(trimmed.to_string());
        }
    }

    if !current_section_lines.is_empty() {
        let desc = current_section_lines.join("\n").trim().to_string();
        if !desc.is_empty() {
            let is_root = current_section_name == doc_root_name;
            let mut relations = Vec::new();
            if !is_root {
                if let (3, Some(ref h2)) = (current_section_level, &current_h2) {
                    relations.push((h2.clone(), "SUBSECTION_OF".to_string(), 0.90));
                } else {
                    relations.push((doc_root_name.clone(), "SECTION_OF".to_string(), 0.95));
                }
            }

            symbols.push(ExtractedSymbol {
                name: current_section_name,
                symbol_type: if is_root {
                    "Document".to_string()
                } else {
                    "DocumentationSection".to_string()
                },
                description: format!(
                    "Documentation in {}:{}. Content:\n{}",
                    file_path, current_section_start, desc
                ),
                file_path: file_path.to_string(),
                line_number: current_section_start,
                parent_symbol: if is_root { None } else { Some(doc_root_name) },
                body: None,
                relations,
            });
        }
    }

    symbols
}

/// Second-pass cross-linking resolver for function calls (CALLS) and type dependencies (USES_TYPE).
pub fn resolve_call_graphs_and_type_dependencies(symbols: &mut [ExtractedSymbol]) {
    // 1. Build lookup tables for declared types and functions
    let mut known_types: HashMap<String, String> = HashMap::new(); // short_name -> full_node_name
    let mut known_functions: HashMap<String, String> = HashMap::new(); // short_name -> full_node_name

    for sym in symbols.iter() {
        match sym.symbol_type.as_str() {
            "Struct" | "Enum" | "Trait" | "Class" | "Interface" | "TypeAlias" => {
                let bare_name = sym
                    .name
                    .split([':', '.'])
                    .next_back()
                    .unwrap_or(&sym.name)
                    .to_string();
                known_types.insert(bare_name, sym.name.clone());
                known_types.insert(sym.name.clone(), sym.name.clone());
            }
            "Function" | "Method" => {
                let bare_name = sym
                    .name
                    .split([':', '.'])
                    .next_back()
                    .unwrap_or(&sym.name)
                    .to_string();
                known_functions.insert(bare_name, sym.name.clone());
                known_functions.insert(sym.name.clone(), sym.name.clone());
            }
            _ => {}
        }
    }

    // 2. Scan each function/method body for references and invocations
    for sym in symbols.iter_mut() {
        if sym.symbol_type != "Function" && sym.symbol_type != "Method" {
            continue;
        }

        let body_text = match sym.body {
            Some(ref b) => b.clone(),
            None => sym.description.clone(),
        };

        let mut existing_targets: HashSet<String> = sym
            .relations
            .iter()
            .map(|(target, _, _)| target.clone())
            .collect();
        let sym_bare = sym.name.split([':', '.']).next_back().unwrap_or(&sym.name);

        // A. Resolve type dependencies (USES_TYPE)
        for (bare_type, full_type) in &known_types {
            if bare_type == sym_bare || full_type == &sym.name {
                continue;
            }

            // Word boundary match
            if is_word_present(&body_text, bare_type) && existing_targets.insert(full_type.clone())
            {
                sym.relations
                    .push((full_type.clone(), "USES_TYPE".to_string(), 0.85));
            }
        }

        // B. Resolve function calls (CALLS)
        for (bare_fn, full_fn) in &known_functions {
            if bare_fn == sym_bare || full_fn == &sym.name {
                continue;
            }

            // Check if invoked with `func(` or `::func(` or `.func(`
            if is_function_called(&body_text, bare_fn) && existing_targets.insert(full_fn.clone()) {
                sym.relations
                    .push((full_fn.clone(), "CALLS".to_string(), 0.85));
            }
        }
    }
}

fn is_word_present(text: &str, word: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0
            || !text[..abs_pos]
                .chars()
                .last()
                .unwrap_or(' ')
                .is_alphanumeric();
        let after_pos = abs_pos + word.len();
        let after_ok = after_pos >= text.len()
            || !text[after_pos..]
                .chars()
                .next()
                .unwrap_or(' ')
                .is_alphanumeric();

        if before_ok && after_ok {
            return true;
        }
        start = abs_pos + word.len();
    }
    false
}

fn is_function_called(text: &str, func_name: &str) -> bool {
    let call_pattern = format!("{}(", func_name);
    let call_pattern_ws = format!("{} (", func_name);
    text.contains(&call_pattern) || text.contains(&call_pattern_ws)
}

fn extract_block_body(lines: &[&str], start_idx: usize) -> String {
    let mut depth = 0;
    let mut body_lines = Vec::new();
    let mut started = false;

    for &line in &lines[start_idx..] {
        let opens = line.matches('{').count();
        let closes = line.matches('}').count();

        if opens > 0 {
            started = true;
        }

        depth += opens;
        depth = depth.saturating_sub(closes);
        body_lines.push(line);

        if started && depth == 0 {
            break;
        }
    }

    body_lines.join("\n")
}

fn extract_python_body(lines: &[&str], start_idx: usize, base_indent: usize) -> String {
    let mut body_lines = Vec::new();
    for &line in &lines[start_idx..] {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            body_lines.push(line);
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if body_lines.is_empty() || indent > base_indent {
            body_lines.push(line);
        } else {
            break;
        }
    }
    body_lines.join("\n")
}

fn extract_keyword_name(line: &str, keyword: &str) -> String {
    let mut after_keyword = false;
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in parts {
        if after_keyword {
            return part
                .split(['<', '{', ':', '(', ';'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
        }
        if part == keyword {
            after_keyword = true;
        }
    }
    String::new()
}

fn extract_function_name(line: &str) -> String {
    let cleaned = line
        .replace("pub ", "")
        .replace("async ", "")
        .replace("const ", "")
        .replace("unsafe ", "")
        .replace("extern ", "");

    if let Some(pos) = cleaned.find("fn ") {
        let after = &cleaned[pos + 3..];
        return after
            .split(['<', '(', ' '])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_code() {
        let code = r#"
        /// Represents a user in the system.
        pub struct UserModel {
            pub id: u64,
            pub name: String,
        }

        impl UserModel {
            pub fn new(name: String) -> Self {
                Self { id: 1, name }
            }
        }

        #[get("/api/users")]
        pub async fn list_users() {}
        "#;

        let symbols = parse_rust_file(code, "src/models/user.rs");
        assert_eq!(symbols.len(), 4);
        assert_eq!(symbols[0].name, "UserModel");
        assert_eq!(symbols[0].symbol_type, "Struct");
        assert_eq!(symbols[1].name, "UserModel::new");
        assert_eq!(symbols[1].symbol_type, "Method");
        assert_eq!(symbols[2].name, "Endpoint get(\"/api/users\")");
        assert_eq!(symbols[2].symbol_type, "Endpoint");
        assert_eq!(symbols[3].name, "list_users");
        assert_eq!(symbols[3].symbol_type, "Function");
    }

    #[test]
    fn test_parse_python_code() {
        let code = r#"
        class AuthService:
            def __init__(self):
                pass
            def validate_token(self, token: str):
                return True

        def global_helper():
            pass
        "#;

        let symbols = parse_python_file(code, "services/auth.py");
        assert_eq!(symbols.len(), 4);
        assert_eq!(symbols[0].name, "AuthService");
        assert_eq!(symbols[1].name, "AuthService.__init__");
        assert_eq!(symbols[2].name, "AuthService.validate_token");
        assert_eq!(symbols[3].name, "global_helper");
    }

    #[test]
    fn test_parse_markdown_document() {
        let md = r#"# Architecture Guide
This is the main architecture documentation of the system.

## Database Storage
We store data in binary zero-copy format.

### Soft Delete Policy
Tasks are never deleted physically.
"#;

        let symbols = parse_markdown_file(md, "docs/ARCHITECTURE.md");
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].name, "ARCHITECTURE.md: Architecture Guide");
        assert_eq!(symbols[0].symbol_type, "DocumentationSection");
        assert_eq!(symbols[1].name, "ARCHITECTURE.md: Database Storage");
        assert_eq!(symbols[1].symbol_type, "DocumentationSection");
        assert_eq!(symbols[2].name, "ARCHITECTURE.md: Soft Delete Policy");
        assert_eq!(symbols[2].symbol_type, "DocumentationSection");
    }

    #[test]
    fn test_call_graph_and_type_dependency_resolution() {
        let code = r#"
        pub struct TaskModel {
            pub id: u64,
            pub title: String,
        }

        pub fn validate_task_title(title: &str) -> bool {
            title.len() >= 3
        }

        pub fn create_task_handler(title: String) -> TaskModel {
            if validate_task_title(&title) {
                TaskModel { id: 1, title }
            } else {
                panic!("Invalid title");
            }
        }
        "#;

        let mut symbols = parse_rust_file(code, "src/handlers/task.rs");
        resolve_call_graphs_and_type_dependencies(&mut symbols);

        let create_task = symbols
            .iter()
            .find(|s| s.name == "create_task_handler")
            .unwrap();

        let has_calls = create_task
            .relations
            .iter()
            .any(|(tgt, rel, _)| tgt == "validate_task_title" && rel == "CALLS");
        let has_uses_type = create_task
            .relations
            .iter()
            .any(|(tgt, rel, _)| tgt == "TaskModel" && rel == "USES_TYPE");

        assert!(
            has_calls,
            "create_task_handler should have CALLS relation to validate_task_title"
        );
        assert!(
            has_uses_type,
            "create_task_handler should have USES_TYPE relation to TaskModel"
        );
    }
}
