mod args;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init(args) => {
            println!("Inicializando banco: {:?} com dim={}", cli.db_path, args.dim);
        }
        Commands::InsertNode(args) => {
            println!("Inserindo nó: '{}'", args.name);
        }
        Commands::InsertEdge(args) => {
            println!("Inserindo aresta: '{}' -> '{}' ({})", args.source, args.target, args.relation);
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
