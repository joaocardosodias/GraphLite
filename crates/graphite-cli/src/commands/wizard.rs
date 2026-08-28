use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

use graphite::engine::config::GraphiteConfig;
use graphite::vector::distance::Metric;
use graphite::vector::embedding::{EmbeddingModelType, LocalEmbedder};
use graphite::vector::quantization::Quantization;
use graphite::vector::reranker::{LocalReranker, RerankerModelType};
use inquire::ui::{Color, RenderConfig, StyleSheet, Styled};
use inquire::{Confirm, CustomType, Select, Text};

fn get_render_config() -> RenderConfig<'static> {
    let mut config = RenderConfig::default_colored();
    config.prompt_prefix = Styled::new("› ").with_fg(Color::LightCyan);
    config.highlighted_option_prefix = Styled::new("› ").with_fg(Color::LightCyan);
    config.selected_option = Some(StyleSheet::new().with_fg(Color::LightCyan));
    config.help_message = StyleSheet::new().with_fg(Color::DarkGrey);
    config.answer = StyleSheet::new().with_fg(Color::White);
    config
}

/// Runs the interactive terminal setup wizard to configure and initialize a Graphite database.
pub fn run_interactive_wizard(default_path: &Path) -> Result<(PathBuf, GraphiteConfig)> {
    let render_config = get_render_config();

    // 1. Target file path with automatic .graph extension
    let default_path_str = default_path
        .to_str()
        .unwrap_or("graphite.graph")
        .to_string();

    let path_input = Text::new("Database path:")
        .with_default(&default_path_str)
        .with_render_config(render_config)
        .prompt()?;

    let mut raw_path = path_input.trim().to_string();
    if !raw_path.is_empty() && !raw_path.ends_with(".graph") && !raw_path.ends_with(".graphite") {
        raw_path.push_str(".graph");
    }

    let db_path = PathBuf::from(raw_path);

    if db_path.exists() {
        let overwrite = Confirm::new("File already exists. Overwrite?")
            .with_default(false)
            .with_render_config(render_config)
            .prompt()?;
        if !overwrite {
            bail!("Initialization aborted: database file already exists.");
        }
    }

    // 2. Embedding Model
    let embedding_choices = [
        EmbeddingModelType::AllMiniLML6V2,
        EmbeddingModelType::MultilingualMiniLML12V2,
        EmbeddingModelType::MultilingualE5Base,
        EmbeddingModelType::BGEM3,
        EmbeddingModelType::BGESmallENV15,
        EmbeddingModelType::NomicEmbedTextV15,
        EmbeddingModelType::Custom(384),
    ];

    let embedding_labels: Vec<String> = embedding_choices
        .iter()
        .map(|m| {
            if *m == EmbeddingModelType::Custom(384) {
                m.display_label().to_string()
            } else if m.is_cached() {
                format!("{}  [Cached]", m.display_label())
            } else {
                m.display_label().to_string()
            }
        })
        .collect();

    let selected_emb_idx = Select::new("Embedding model:", embedding_labels)
        .with_render_config(render_config)
        .raw_prompt()?
        .index;

    let selected_emb = embedding_choices[selected_emb_idx];

    let dim = if let EmbeddingModelType::Custom(_) = selected_emb {
        CustomType::<usize>::new("Vector dimension:")
            .with_default(384)
            .with_render_config(render_config)
            .prompt()?
    } else {
        selected_emb.dimension()
    };

    // 3. Reranking Model (always required for maximum retrieval precision)
    let reranker_choices = [
        RerankerModelType::BGERerankerBase,
        RerankerModelType::JinaRerankerV2BaseMultilingual,
        RerankerModelType::BGERerankerV2M3,
        RerankerModelType::JinaRerankerV1TurboEn,
    ];

    let reranker_labels: Vec<String> = reranker_choices
        .iter()
        .map(|r| {
            if r.is_cached() {
                format!("{}  [Cached]", r.display_label())
            } else {
                r.display_label().to_string()
            }
        })
        .collect();

    let selected_rerank_idx = Select::new("Reranking model:", reranker_labels)
        .with_render_config(render_config)
        .raw_prompt()?
        .index;

    let selected_rerank = reranker_choices[selected_rerank_idx];

    // 4. Distance Metric
    let metric_options = vec![
        "Cosine Similarity       (Recommended for dense vectors)",
        "Dot Product             (Normalized embeddings)",
        "Euclidean Distance      (L2)",
        "Manhattan Distance      (L1)",
    ];

    let metric_idx = Select::new("Distance metric:", metric_options)
        .with_render_config(render_config)
        .raw_prompt()?
        .index;

    let metric = match metric_idx {
        0 => Metric::Cosine,
        1 => Metric::DotProduct,
        2 => Metric::Euclidean,
        _ => Metric::Manhattan,
    };

    // 5. Quantization
    let quant_options = vec![
        "ScalarInt8 (SQ8)        (Recommended: 75% smaller, SIMD accelerated)",
        "Float32                 (Full 32-bit decimal precision)",
    ];

    let quant_idx = Select::new("Storage quantization:", quant_options)
        .with_render_config(render_config)
        .raw_prompt()?
        .index;

    let quantization = match quant_idx {
        0 => Quantization::ScalarInt8,
        _ => Quantization::None,
    };

    // 6. Hardware Acceleration (CPU vs CUDA)
    let cuda_status = graphite::vector::device::CudaStatus::detect();
    let auto_label = match &cuda_status {
        graphite::vector::device::CudaStatus::Available { device_count } => {
            format!(
                "Auto (Recommended: {} NVIDIA GPU(s) detected with CUDA)",
                device_count
            )
        }
        graphite::vector::device::CudaStatus::GpuDetectedDriverMissing { .. } => {
            "Auto (NVIDIA GPU detected, driver missing -> runs on CPU)".to_string()
        }
        graphite::vector::device::CudaStatus::NoGpuDetected => {
            "Auto (CPU SIMD AVX2/AVX-512 acceleration)".to_string()
        }
    };

    let device_options = vec![
        auto_label,
        "CPU (Force standard multi-threaded SIMD execution)".to_string(),
        "CUDA (Force NVIDIA GPU Tensor Cores execution)".to_string(),
    ];

    let device_idx = Select::new("Hardware acceleration:", device_options)
        .with_render_config(render_config)
        .raw_prompt()?
        .index;

    let device = match device_idx {
        1 => graphite::vector::DeviceType::Cpu,
        2 => graphite::vector::DeviceType::Cuda(0),
        _ => graphite::vector::DeviceType::Auto,
    };

    // If GPU is detected but driver is missing, offer installation prompt
    if let graphite::vector::device::CudaStatus::GpuDetectedDriverMissing {
        ref distro_id,
        ref install_command,
    } = cuda_status
    {
        if device == graphite::vector::DeviceType::Auto || device.is_cuda() {
            println!();
            println!("  [!] NVIDIA GPU detected on system, but CUDA runtime is not active.");
            println!("      To enable full GPU acceleration on {}:", distro_id);
            println!("        {}", install_command);
            let should_install = Confirm::new("Would you like to run the installation command now?")
                .with_default(false)
                .with_render_config(render_config)
                .prompt()
                .unwrap_or(false);

            if should_install {
                println!("\nExecuting: {}\n", install_command);
                let _ = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(install_command)
                    .status();
            } else {
                println!("  Graphite will proceed smoothly on CPU with SIMD acceleration.\n");
            }
        }
    }

    // 7. Pre-download / Cache verification
    let emb_cached = selected_emb.is_cached();
    let rerank_cached = selected_rerank.is_cached();

    if !emb_cached || !rerank_cached {
        let download_confirmed = Confirm::new("Download missing ONNX model weights now?")
            .with_default(true)
            .with_render_config(render_config)
            .prompt()?;

        if download_confirmed {
            println!();
            if !emb_cached {
                if let EmbeddingModelType::Custom(_) = selected_emb {
                    // No download needed
                } else {
                    println!("  Downloading embedding model: {}...", selected_emb.name());
                    LocalEmbedder::from_model_type_and_device(selected_emb, device)?;
                    println!("  Embedding model ready.");
                }
            }

            if !rerank_cached && selected_rerank != RerankerModelType::None {
                println!(
                    "  Downloading reranker model: {}...",
                    selected_rerank.name()
                );
                LocalReranker::from_model_type_and_device(selected_rerank, device)?;
                println!("  Reranker model ready.");
            }
            println!();
        }
    }

    let config = GraphiteConfig::new()
        .with_dim(dim)
        .with_metric(metric)
        .with_quantization(quantization)
        .with_models(selected_emb.id(), selected_rerank.id())
        .with_device(device)
        .with_auto_flush(true);

    Ok((db_path, config))
}
