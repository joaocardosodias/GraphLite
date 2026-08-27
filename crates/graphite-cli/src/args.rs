use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Graphite: An embedded, single-file Graph + Vector / GraphRAG engine in pure Rust.
#[derive(Parser, Debug)]
#[command(
    name = "graphite",
    author,
    version,
    about = "Embedded single-file Graph + Vector / GraphRAG engine in pure Rust",
    long_about = "Graphite is a lightning-fast, local-first embedded database combining Compressed Sparse Row (CSR) graph traversal with SIMD-accelerated Int8 quantized vector search."
)]
pub struct Cli {
    /// Path to the `.graph` database file.
    #[arg(
        short = 'd',
        long = "db",
        default_value = "graphite.graphite",
        global = true,
        help = "Path to the target .graph database file"
    )]
    pub db_path: PathBuf,

    /// Enable verbose diagnostic output.
    #[arg(short, long, global = true, help = "Enable verbose debug logging")]
    pub verbose: bool,

    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize or create a new `.graph` database file with interactive setup wizard.
    #[command(alias = "create")]
    Init(InitArgs),

    /// Create a new `.graph` database file (alias for init).
    Create(InitArgs),

    /// Insert or update an entity node with optional embedding vector.
    InsertNode(InsertNodeArgs),

    /// Insert a relational edge connecting two entities.
    InsertEdge(InsertEdgeArgs),

    /// Execute a GraphRAG context retrieval query or vector search.
    Query(QueryArgs),

    /// Inspect database statistics, binary header, and memory layout.
    Inspect(InspectArgs),

    /// Export database content and topology as JSON or Markdown triples.
    Dump(DumpArgs),

    /// Ingest documents (Markdown, PDF, Text, JSON, CSV) with hierarchical semantic chunking into the knowledge graph.
    Ingest(IngestArgs),

    /// Record an agent memory, fact, rule, or preference with automatic embedding and relational linking.
    Remember(RememberArgs),

    /// Launch embedded HTTP / REST API server for Python, Node.js, and web clients.
    Serve(ServeArgs),
}

#[derive(Args, Debug, Clone)]
pub struct InitArgs {
    /// Local embedding model to use (e.g. all-minilm-l6-v2, multilingual-minilm-l12-v2, multilingual-e5-base, bge-m3, nomic-embed-text-v1.5, custom).
    #[arg(
        short = 'e',
        long = "embedding-model",
        help = "Embedding model (dimension is inferred automatically)"
    )]
    pub embedding_model: Option<String>,

    /// Local reranker model to use (e.g. bge-reranker-base, bge-reranker-v2-m3, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual, none).
    #[arg(
        short = 'r',
        long = "reranker-model",
        help = "Reranker model (cross-encoder for deep accuracy, or 'none')"
    )]
    pub reranker_model: Option<String>,

    /// Launch interactive terminal wizard to configure the database.
    #[arg(
        short = 'i',
        long = "interactive",
        help = "Launch interactive terminal setup wizard"
    )]
    pub interactive: bool,

    /// Immediately pre-download and verify ONNX model weights with progress bar.
    #[arg(
        long = "download",
        help = "Pre-download and cache model weights immediately"
    )]
    pub download: bool,

    /// Vector embedding dimensionality (inferred automatically from the chosen embedding model).
    #[arg(
        short = 'D',
        long,
        help = "Vector dimension (only needed for custom models without local embedder)"
    )]
    pub dim: Option<usize>,

    /// Vector distance metric.
    #[arg(short = 'm', long, value_enum, default_value_t = CliMetric::Cosine, help = "Distance metric")]
    pub metric: CliMetric,

    /// Vector quantization mode.
    #[arg(short = 'q', long, value_enum, default_value_t = CliQuantization::ScalarInt8, help = "Quantization mode")]
    pub quantization: CliQuantization,

    /// Default token budget allocated for LLM context retrieval.
    #[arg(
        short = 't',
        long,
        default_value_t = 2048,
        help = "Default token budget for prompts"
    )]
    pub max_tokens: usize,

    /// Overwrite existing database file if it already exists.
    #[arg(short = 'f', long, help = "Overwrite existing database file")]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct InsertNodeArgs {
    /// Name/label of the entity node.
    #[arg(short = 'n', long, help = "Entity name / label (e.g. 'Projeto Titan')")]
    pub name: String,

    /// Category or type of entity.
    #[arg(
        short = 't',
        long,
        default_value = "",
        help = "Entity type (e.g. 'Person', 'Project')"
    )]
    pub entity_type: String,

    /// Text description / summary of the entity.
    #[arg(
        short = 'D',
        long,
        default_value = "",
        help = "Textual summary or content"
    )]
    pub description: String,

    /// Comma-separated float values of the embedding vector, or path to a JSON file.
    #[arg(
        short = 'V',
        long,
        help = "Comma-separated vector floats (e.g. '0.1,0.2,0.3')"
    )]
    pub vector: Option<String>,

    /// Automatically compute vector embedding in pure Rust from name and description.
    #[arg(long = "auto-embed", help = "Automatically compute vector embedding")]
    pub auto_embed: bool,

    /// Automatically merge with an existing node if semantic cosine similarity >= 0.92.
    #[arg(
        long,
        default_value_t = true,
        help = "Enable real-time entity resolution"
    )]
    pub resolve: bool,
}

#[derive(Args, Debug)]
pub struct InsertEdgeArgs {
    /// Name of the source entity node.
    #[arg(short = 's', long, help = "Name of the source node")]
    pub source: String,

    /// Name of the target entity node.
    #[arg(short = 't', long, help = "Name of the target node")]
    pub target: String,

    /// Relationship label (e.g. 'LEADS', 'USES', 'DEPENDS_ON').
    #[arg(short = 'r', long, help = "Relation type label")]
    pub relation: String,

    /// Semantic confidence weight of the connection (0.0 to 1.0).
    #[arg(
        short = 'w',
        long,
        default_value_t = 1.0,
        help = "Edge weight between 0.0 and 1.0"
    )]
    pub weight: f32,

    /// Whether the relationship is directed (source -> target).
    #[arg(long, default_value_t = true, help = "Whether edge is directed")]
    pub directed: bool,
}

#[derive(Args, Debug)]
pub struct QueryArgs {
    /// Comma-separated query embedding vector floats.
    #[arg(short = 'V', long, help = "Comma-separated query vector floats")]
    pub vector: Option<String>,

    /// Plain text search query (computes vector embedding automatically in pure Rust).
    #[arg(
        short = 'T',
        long = "text",
        help = "Plain text search query (auto-embedded)"
    )]
    pub query_text: Option<String>,

    /// Comma-separated seed entity names for textual exploration.
    #[arg(
        short = 's',
        long,
        help = "Comma-separated seed entity names (e.g. 'Titan,Ana')"
    )]
    pub seeds: Option<String>,

    /// Number of seed entities to retrieve via vector search.
    #[arg(short = 'k', long, default_value_t = 5, help = "Top-K seed entries")]
    pub top_k: usize,

    /// Maximum token budget for the returned prompt.
    #[arg(
        short = 't',
        long,
        help = "Token budget limit (overrides config default)"
    )]
    pub tokens: Option<usize>,

    /// Maximum BFS graph exploration depth in hops.
    #[arg(long, help = "Maximum BFS search depth in hops (e.g. 1 or 2)")]
    pub depth: Option<usize>,

    /// Alpha balancing factor between vector score and graph topology ($0.0 \le \alpha \le 1.0$).
    #[arg(
        long,
        help = "Hybrid score alpha (1.0 = pure vector, 0.0 = pure graph)"
    )]
    pub alpha: Option<f32>,

    /// Filter retrieved entities by comma-separated entity types (e.g. 'Function,Struct,Class,DatabaseTable').
    #[arg(
        short = 'y',
        long = "type",
        help = "Filter by entity type (e.g. 'Function,Struct')"
    )]
    pub entity_type: Option<String>,

    /// Output formatting mode.
    #[arg(short = 'f', long, value_enum, default_value_t = CliOutputFormat::Markdown, help = "Output format")]
    pub format: CliOutputFormat,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Display full JSON dump of header metadata.
    #[arg(long, help = "Output inspection data as JSON")]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct DumpArgs {
    /// Format of the exported graph.
    #[arg(short = 'f', long, value_enum, default_value_t = CliDumpFormat::Json, help = "Export format")]
    pub format: CliDumpFormat,
}

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// Path to the file or directory containing documents to ingest.
    #[arg(default_value = ".", help = "Path to file or directory to ingest")]
    pub path: std::path::PathBuf,

    /// Target token size per semantic text chunk (approx 4 chars per token).
    #[arg(
        short = 's',
        long,
        default_value_t = 350,
        help = "Target token size per chunk"
    )]
    pub chunk_size: usize,

    /// Overlap tokens between adjacent sliding window chunks.
    #[arg(
        short = 'o',
        long,
        default_value_t = 40,
        help = "Overlap tokens between chunks"
    )]
    pub chunk_overlap: usize,

    /// Comma-separated list of file extensions to ingest (e.g. 'md,txt,json,csv,pdf').
    #[arg(short = 'e', long, help = "File extensions to include")]
    pub extensions: Option<String>,

    /// Maximum number of files to ingest.
    #[arg(
        long,
        default_value_t = 1000,
        help = "Maximum number of files to ingest"
    )]
    pub max_files: usize,

    /// Continuous watch mode: auto-reingest when files are modified or added.
    #[arg(
        short = 'w',
        long,
        help = "Watch directory for changes and continuously re-index"
    )]
    pub watch: bool,

    /// Force re-indexing of all files ignoring cached content hashes.
    #[arg(
        short = 'f',
        long,
        help = "Force re-indexing ignoring cached file hashes"
    )]
    pub force: bool,

    /// Write directly to database file without creating temporary staging files (.tmp).
    #[arg(
        long,
        help = "Direct write mode without temporary staging files (.tmp)"
    )]
    pub no_tmp: bool,
}

#[derive(Args, Debug)]
pub struct RememberArgs {
    /// The fact, preference, rule, or memory text to record.
    #[arg(help = "Memory text content to record")]
    pub text: String,

    /// Optional semantic category or type for the memory (e.g. 'preference', 'fact', 'task', 'rule').
    #[arg(
        short = 'c',
        long,
        default_value = "AgentMemory",
        help = "Category/type label for the memory"
    )]
    pub category: String,

    /// Optional related entity name to establish an immediate connection with.
    #[arg(
        short = 'r',
        long,
        help = "Name of an existing related entity to connect to"
    )]
    pub relate_to: Option<String>,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Host address to bind the HTTP server to.
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "Host IP address to bind server"
    )]
    pub host: String,

    /// Port number to listen on.
    #[arg(
        short = 'p',
        long,
        default_value_t = 8000,
        help = "Port number to listen on"
    )]
    pub port: u16,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliMetric {
    Cosine,
    DotProduct,
    Euclidean,
    Manhattan,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliQuantization {
    None,
    ScalarInt8,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliOutputFormat {
    Markdown,
    Json,
    Triples,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliDumpFormat {
    Json,
    Triples,
    Markdown,
}
