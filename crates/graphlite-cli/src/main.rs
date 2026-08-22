mod args;
mod commands;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Commands};
use commands::dump::execute_dump;
use commands::init::execute_init;
use commands::insert::{execute_insert_edge, execute_insert_node};
use commands::inspect::execute_inspect;
use commands::query::execute_query;

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
    }

    Ok(())
}
