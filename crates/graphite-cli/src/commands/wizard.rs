use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

use graphite::engine::config::GraphiteConfig;
use graphite::vector::distance::Metric;
use graphite::vector::embedding::{EmbeddingModelType, LocalEmbedder};
use graphite::vector::quantization::Quantization;
use graphite::vector::reranker::{LocalReranker, RerankerModelType};
use inquire::{Confirm, CustomType, Select, Text};

/// Runs the interactive terminal setup wizard to configure and initialize a Graphite database.
pub fn run_interactive_wizard(default_path: &Path) -> Result<(PathBuf, GraphiteConfig)> {
    println!();
    println!("=== Assistente de Criação do Graphite DB ===");
    println!("Configure o seu banco de dados vetorial e grafo de conhecimento.");
    println!();

    // 1. Caminho do arquivo
    let default_path_str = default_path
        .to_str()
        .unwrap_or("knowledge.graphite")
        .to_string();

    let path_input = Text::new("Caminho do arquivo do banco (.graphite):")
        .with_default(&default_path_str)
        .with_help_message("Pressione Enter para usar o caminho padrão ou digite um novo")
        .prompt()?;

    let db_path = PathBuf::from(path_input.trim());

    if db_path.exists() {
        let overwrite = Confirm::new("O arquivo já existe. Deseja sobrescrever?")
            .with_default(false)
            .prompt()?;
        if !overwrite {
            bail!("Criação cancelada: o arquivo já existe.");
        }
    }

    // 2. Modelo de Embedding
    let embedding_choices = vec![
        EmbeddingModelType::AllMiniLML6V2,
        EmbeddingModelType::MultilingualMiniLML12V2,
        EmbeddingModelType::MultilingualE5Base,
        EmbeddingModelType::BGEM3,
        EmbeddingModelType::BGESmallENV15,
        EmbeddingModelType::NomicEmbedTextV15,
        EmbeddingModelType::Custom(384),
    ];

    let embedding_labels: Vec<&str> = embedding_choices
        .iter()
        .map(|m| m.display_label())
        .collect();

    let selected_emb_idx = Select::new(
        "Selecione o Modelo de Embedding (Vetorização):",
        embedding_labels,
    )
    .with_help_message("Use as setas para cima/baixo e Enter para selecionar")
    .raw_prompt()?
    .index;

    let selected_emb = embedding_choices[selected_emb_idx];

    let dim = if let EmbeddingModelType::Custom(_) = selected_emb {
        CustomType::<usize>::new("Digite a dimensão manual dos vetores:")
            .with_default(384)
            .prompt()?
    } else {
        selected_emb.dimension()
    };

    // 3. Modelo de Reranking
    let reranker_choices = vec![
        RerankerModelType::BGERerankerBase,
        RerankerModelType::JinaRerankerV2BaseMultilingual,
        RerankerModelType::BGERerankerV2M3,
        RerankerModelType::JinaRerankerV1TurboEn,
        RerankerModelType::None,
    ];

    let reranker_labels: Vec<&str> = reranker_choices.iter().map(|r| r.display_label()).collect();

    let selected_rerank_idx = Select::new(
        "Selecione o Modelo de Reranking (Reclassificador Neural):",
        reranker_labels,
    )
    .with_help_message("O Reranker aplica atenção cruzada para máxima precisão semântica")
    .raw_prompt()?
    .index;

    let selected_rerank = reranker_choices[selected_rerank_idx];

    // 4. Métrica de Distância
    let metric_options = vec![
        "Cosine (Similaridade de Cosseno) - Recomendado para Embeddings",
        "DotProduct (Produto Escalar)",
        "Euclidean (Distância Euclidiana L2)",
        "Manhattan (Distância de Manhattan L1)",
    ];

    let metric_idx = Select::new("Métrica de Distância:", metric_options)
        .raw_prompt()?
        .index;

    let metric = match metric_idx {
        0 => Metric::Cosine,
        1 => Metric::DotProduct,
        2 => Metric::Euclidean,
        _ => Metric::Manhattan,
    };

    // 5. Quantização
    let quant_options = vec![
        "ScalarInt8 (SQ8 - Redução de 75% de memória e disco) - Recomendado",
        "Float32 (Sem quantização, precisão decimal completa)",
    ];

    let quant_idx = Select::new("Modo de Armazenamento Vetorial:", quant_options)
        .raw_prompt()?
        .index;

    let quantization = match quant_idx {
        0 => Quantization::ScalarInt8,
        _ => Quantization::None,
    };

    // 6. Pré-download
    let pre_download = if selected_emb != EmbeddingModelType::Custom(dim)
        || selected_rerank != RerankerModelType::None
    {
        Confirm::new("Deseja baixar e verificar os pesos dos modelos agora?")
            .with_default(true)
            .with_help_message("Garante que os modelos estejam prontos para uso offline")
            .prompt()?
    } else {
        false
    };

    if pre_download {
        println!();
        if let EmbeddingModelType::Custom(_) = selected_emb {
            // No local download needed
        } else {
            println!(
                "Verificando / Baixando modelo de Embedding: {}...",
                selected_emb.name()
            );
            LocalEmbedder::from_model_type(selected_emb)?;
            println!("Modelo de Embedding pronto.");
        }

        if selected_rerank != RerankerModelType::None {
            println!(
                "Verificando / Baixando modelo de Reranking: {}...",
                selected_rerank.name()
            );
            LocalReranker::from_model_type(selected_rerank)?;
            println!("Modelo de Reranking pronto.");
        }
        println!();
    }

    let config = GraphiteConfig::new()
        .with_dim(dim)
        .with_metric(metric)
        .with_quantization(quantization)
        .with_models(selected_emb.id(), selected_rerank.id())
        .with_auto_flush(true);

    Ok((db_path, config))
}
