mod args;
mod commands;
pub mod ingestion;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Commands};
use commands::dump::execute_dump;
use commands::init::execute_init;
use commands::insert::{execute_insert_edge, execute_insert_node};
use commands::inspect::execute_inspect;
use commands::query::execute_query;
use commands::remember::execute_remember;
use commands::serve::execute_serve;
use ingestion::execute_ingest;

fn normalize_db_path(path: &std::path::Path) -> std::path::PathBuf {
    let s = path.to_string_lossy();
    if !s.ends_with(".graph") && !s.ends_with(".graphite") {
        let mut p = path.to_path_buf();
        p.set_extension("graph");
        p
    } else {
        path.to_path_buf()
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = normalize_db_path(&cli.db_path);

    match &cli.command {
        Commands::Init(args) | Commands::Create(args) => {
            let mut init_args = args.clone();
            if init_args.embedding_model.is_none() && init_args.dim.is_none() {
                init_args.interactive = true;
            }
            execute_init(&db_path, &init_args)?;
        }
        Commands::InsertNode(args) => {
            execute_insert_node(&db_path, args)?;
        }
        Commands::InsertEdge(args) => {
            execute_insert_edge(&db_path, args)?;
        }
        Commands::Query(args) => {
            execute_query(&db_path, args, cli.verbose)?;
        }
        Commands::Inspect(args) => {
            execute_inspect(&db_path, args)?;
        }
        Commands::Dump(args) => {
            execute_dump(&db_path, args)?;
        }
        Commands::Ingest(args) => {
            execute_ingest(&db_path, args)?;
        }
        Commands::Remember(args) => {
            execute_remember(&db_path, args)?;
        }
        Commands::Serve(args) => {
            execute_serve(&db_path, args)?;
        }
        Commands::Doctor(args) => {
            commands::doctor::execute_doctor(args)?;
        }
    }

    Ok(())
}
