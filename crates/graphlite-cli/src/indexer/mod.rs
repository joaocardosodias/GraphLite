//! Codebase AST & Symbol Indexer for Rust, Python, TypeScript, and Go.

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
            && !trimmed.contains(";")
        {
            let name = extract_keyword_name(trimmed, "struct");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
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
                    relations: Vec::new(),
                });
                current_docs.clear();
            }
        }

        // Match `pub enum Foo` or `enum Foo`
        if (trimmed.starts_with("pub enum ") || trimmed.starts_with("enum "))
            && !trimmed.contains(";")
        {
            let name = extract_keyword_name(trimmed, "enum");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
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
                    relations: Vec::new(),
                });
                current_docs.clear();
            }
        }

        // Match `pub trait Foo` or `trait Foo`
        if (trimmed.starts_with("pub trait ") || trimmed.starts_with("trait "))
            && !trimmed.contains(";")
        {
            let name = extract_keyword_name(trimmed, "trait");
            if !name.is_empty() {
                let doc_summary = current_docs.join(" ");
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
                    symbol_type: "Function".to_string(),
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

    for (idx, line) in content.lines().enumerate() {
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

    for (idx, line) in content.lines().enumerate() {
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

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("type ") && trimmed.contains("struct") {
            let name = trimmed
                .trim_start_matches("type ")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                symbols.push(ExtractedSymbol {
                    name: name.to_string(),
                    symbol_type: "Struct".to_string(),
                    description: format!("Go Struct '{}' in {}:{}", name, file_path, idx + 1),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
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
                symbols.push(ExtractedSymbol {
                    name: name.to_string(),
                    symbol_type: "Interface".to_string(),
                    description: format!("Go Interface '{}' in {}:{}", name, file_path, idx + 1),
                    file_path: file_path.to_string(),
                    line_number: idx + 1,
                    parent_symbol: None,
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
                        relations: vec![(receiver_type.to_string(), "METHOD_OF".to_string(), 0.95)],
                    });
                }
            } else {
                let func_name = func_part.split('(').next().unwrap_or("").trim();
                if !func_name.is_empty() {
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
                        relations: Vec::new(),
                    });
                }
            }
        }
    }

    symbols
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
        assert_eq!(symbols[1].symbol_type, "Function");
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
}
