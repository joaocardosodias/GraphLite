use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "graphlite", author, version, about = "Embedded GraphRAG and Vector Engine in pure Rust")]
struct Cli {
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    println!("GraphLite CLI v{}", graphlite_core::VERSION);
    Ok(())
}
