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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init(args) => {
            execute_init(&cli.db_path, args)?;
        }
        Commands::InsertNode(args) => {
            execute_insert_node(&cli.db_path, args)?;
        }
        Commands::InsertEdge(args) => {
            execute_insert_edge(&cli.db_path, args)?;
        }
        Commands::Query(args) => {
            execute_query(&cli.db_path, args, cli.verbose)?;
        }
        Commands::Inspect(args) => {
            execute_inspect(&cli.db_path, args)?;
        }
        Commands::Dump(args) => {
            execute_dump(&cli.db_path, args)?;
        }
        Commands::Ingest(args) => {
            execute_ingest(&cli.db_path, args)?;
        }
        Commands::Remember(args) => {
            execute_remember(&cli.db_path, args)?;
        }
        Commands::Serve(args) => {
            execute_serve(&cli.db_path, args)?;
        }
    }

    Ok(())
}
