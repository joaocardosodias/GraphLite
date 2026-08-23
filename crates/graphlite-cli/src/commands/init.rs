use anyhow::{bail, Result};
use std::fs;
use std::path::Path;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliMetric, CliQuantization, InitArgs};

pub fn execute_init(db_path: &Path, args: &InitArgs) -> Result<()> {
    if db_path.exists() {
        if !args.force {
            bail!(
                "Database file '{:?}' already exists. Use '--force' (-f) to overwrite.",
                db_path
            );
        } else {
            fs::remove_file(db_path)?;
        }
    }

    let metric = match args.metric {
        CliMetric::Cosine => Metric::Cosine,
        CliMetric::DotProduct => Metric::DotProduct,
        CliMetric::Euclidean => Metric::Euclidean,
        CliMetric::Manhattan => Metric::Manhattan,
    };

    let quantization = match args.quantization {
        CliQuantization::None => Quantization::None,
        CliQuantization::ScalarInt8 => Quantization::ScalarInt8,
    };

    let config = GraphLiteConfig::new()
        .with_dim(args.dim)
        .with_metric(metric)
        .with_quantization(quantization)
        .with_max_tokens(args.max_tokens)
        .with_auto_flush(true);

    let engine = GraphLiteEngine::open_or_create(db_path, config)?;
    engine.flush()?;

    println!("=== GraphLite Database Initialized ===");
    println!("  Database File:        {:?}", db_path);
    println!("  Vector Dimension:     {} dimensions", args.dim);
    println!("  Distance Metric:      {:?}", args.metric);
    println!("  Quantization:         {:?}", args.quantization);
    println!("  Default Token Budget: {}", args.max_tokens);

    let project_dir = if let Some(parent) = db_path.parent() {
        if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        }
    } else {
        Path::new(".")
    };

    let db_filename = db_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("app.graph");

    if !args.no_rules {
        setup_agent_rules(project_dir, db_filename)?;
    }

    println!(
        "\n💡 Dica: Execute 'graphlite -d {} ingest ./documentos' para ingerir arquivos e construir o grafo de conhecimento!",
        db_path.display()
    );

    Ok(())
}

fn setup_agent_rules(project_dir: &Path, db_filename: &str) -> Result<()> {
    let rule_template = format!(
        "# Project Knowledge & AI Agent Memory Directives\n\n\
        This project uses **GraphLite** as its embedded GraphRAG knowledge base and long-term memory engine persisted in `{}`.\n\n\
        ## Mandatory Rules for AI Assistants:\n\
        1. **Knowledge & Context Retrieval:**\n\
           - Whenever answering questions regarding system architecture, business rules, policies, APIs, or user preferences, you **MUST** call the MCP tool `graphlite_retrieve(query: \"...\")`.\n\
        2. **Prohibition of Direct Binary File Reading:**\n\
           - The file `{}` is a compiled single-file binary database. **NEVER** attempt to inspect `{}` using `cat`, `read_file`, `head`, `strings`, or SQLite CLI tools.\n\
           - All knowledge inspection and retrieval must be performed exclusively via the `graphlite_retrieve` MCP tool.\n\
        3. **Continuous Agent Memory & Knowledge Persistence:**\n\
           - When discovering new business rules, domain facts, or user preferences during conversations, persist them using `graphlite_remember(name: \"...\", type: \"...\", description: \"...\")` and connect dependencies via `graphlite_connect`.\n",
        db_filename, db_filename, db_filename
    );

    println!(
        "\nConfiguring AI Assistant rules in '{}'...",
        project_dir.display()
    );

    let target_files = ["AGENTS.md", "CLAUDE.md", "GEMINI.md"];

    for filename in &target_files {
        let file_path = project_dir.join(filename);
        if file_path.exists() {
            let content = fs::read_to_string(&file_path)?;
            if content.contains("graphlite_retrieve") || content.contains("GraphLite") {
                println!(
                    "  [✓] {} - Already configured with GraphLite directives.",
                    filename
                );
            } else {
                let mut updated_content = content.trim_end().to_string();
                updated_content.push_str("\n\n");
                updated_content.push_str(&rule_template);
                fs::write(&file_path, updated_content)?;
                println!("  [+] {} - Appended GraphLite memory directives.", filename);
            }
        } else {
            fs::write(&file_path, &rule_template)?;
            println!(
                "  [+] {} - Created with GraphLite memory directives.",
                filename
            );
        }
    }

    Ok(())
}
