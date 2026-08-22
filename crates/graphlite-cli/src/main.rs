mod args;
mod commands;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Commands};
use commands::init::execute_init;
use commands::insert::{execute_insert_edge, execute_insert_node};

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
        Commands::Query(_args) => {
            println!("Executando query no banco: {:?}", cli.db_path);
        }
        Commands::Inspect(_) => {
            println!("Inspecionando banco: {:?}", cli.db_path);
        }
        Commands::Dump(_) => {
            println!("Exportando banco: {:?}", cli.db_path);
        }
    }

    Ok(())
}
