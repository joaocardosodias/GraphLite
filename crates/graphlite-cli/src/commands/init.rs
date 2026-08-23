use anyhow::{bail, Result};
use std::fs;
use std::path::Path;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliMetric, CliQuantization, IndexCodeArgs, InitArgs};
use crate::commands::index_code::execute_index_code;

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

    if args.index {
        println!(
            "\nAuto-indexing codebase symbols into '{}'...",
            db_path.display()
        );
        let index_args = IndexCodeArgs {
            path: project_dir.to_path_buf(),
            extensions: None,
            max_files: 2000,
        };
        execute_index_code(db_path, &index_args)?;
    } else {
        println!("\n💡 Tip: Run 'graphlite -d {} index-code .' to automatically index structs, functions and routes!", db_path.display());
    }

    Ok(())
}

fn setup_agent_rules(project_dir: &Path, db_filename: &str) -> Result<()> {
    let rule_template = format!(
        "# Diretrizes de Memória e Conhecimento do Projeto\n\n\
        Este projeto utiliza o **GraphLite** como motor de memória arquitetural e grafo de conhecimento persistido em `{}`.\n\n\
        ## Regras Obrigatórias para Assistentes de IA:\n\
        1. **Consulta de Arquitetura e Regras:**\n\
           - Sempre que o usuário fizer perguntas sobre como o sistema funciona, regras de negócio, autenticação, modelos de dados ou endpoints, você **DEVE** acionar a ferramenta MCP `graphlite_retrieve(query: \"...\")`.\n\
        2. **Proibição de Leitura Direta de Binários:**\n\
           - O arquivo `{}` é um banco binário compilado. **NUNCA** tente ler `{}` usando `cat`, `read_file`, `head` ou utilitários SQL como `sqlite3`.\n\
           - Todas as operações de leitura devem ser feitas exclusivamente via `graphlite_retrieve`.\n\
        3. **Persistência de Novas Regras:**\n\
           - Ao aprender novas regras ou preferências do usuário, chame `graphlite_remember` e `graphlite_connect`.\n",
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
