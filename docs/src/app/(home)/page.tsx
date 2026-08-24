import Link from 'next/link';
import { ArrowRight, Terminal, Cpu, Database, Zap, Layers, Network, ShieldCheck } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="relative min-h-[calc(100vh-4rem)] flex flex-col justify-between max-w-6xl mx-auto px-6 py-12 md:py-20">
      
      {/* Hero Section */}
      <section className="text-center max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 mb-6">
          <span>v0.1.0 • Pure Rust • Zero-Copy Mmap</span>
        </div>

        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-white mb-6">
          The Embedded Engine for GraphRAG & AI Memory
        </h1>

        <p className="text-base sm:text-lg text-neutral-400 leading-relaxed mb-8">
          Graphite combines Knowledge Graphs, SIMD AVX2 Vector Search, BM25 Lexical Indexing, and Local Cross-Encoder Reranking into a single, zero-dependency <code className="text-sm font-mono text-emerald-400 bg-neutral-900 px-1.5 py-0.5 rounded border border-neutral-800">.graphite</code> binary file.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-black bg-white hover:bg-neutral-200 transition-colors shadow-sm"
          >
            Documentation
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/docs/quickstart"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-neutral-300 bg-neutral-900 border border-neutral-800 hover:bg-neutral-800 hover:text-white transition-colors"
          >
            Quickstart
          </Link>
          <a
            href="https://github.com/joaocardosodias/Graphite"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-neutral-400 hover:text-neutral-200 transition-colors"
          >
            GitHub
          </a>
        </div>
      </section>

      {/* Terminal Code Preview */}
      <section className="w-full max-w-3xl mx-auto my-12">
        <div className="rounded-xl border border-neutral-800 bg-[#080808] overflow-hidden shadow-xl">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-neutral-800 bg-[#0c0c0c] text-xs font-mono text-neutral-400">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-neutral-700" />
              <div className="w-2.5 h-2.5 rounded-full bg-neutral-700" />
              <div className="w-2.5 h-2.5 rounded-full bg-neutral-700" />
              <span className="ml-2 text-neutral-500">bash</span>
            </div>
            <span className="text-neutral-500">graphite-cli</span>
          </div>
          <div className="p-5 font-mono text-xs sm:text-sm text-neutral-300 space-y-3 overflow-x-auto leading-relaxed">
            <div>
              <span className="text-emerald-400">$</span> cargo install --path crates/graphite-cli
            </div>
            <div>
              <span className="text-emerald-400">$</span> graphite ingest ./documents -d knowledge.graphite --no-tmp
              <div className="text-neutral-500 text-xs mt-0.5">
                Ingested 492 nodes, 980 edges in 47.4s (file: knowledge.graphite)
              </div>
            </div>
            <div>
              <span className="text-emerald-400">$</span> graphite -d knowledge.graphite query -T "artigo 861 do codigo civil" --rerank -v
              <div className="mt-2 pl-3 border-l-2 border-emerald-500/40 text-neutral-400 text-xs space-y-1">
                <div className="text-neutral-200 font-semibold"># Retrieved Knowledge Context (Reranked):</div>
                <div>- [codigo_civil_lei_10406.md: (Part 163)] (Relevance: 0.47)</div>
                <div className="text-neutral-300">  Art. 861. Aquele que, sem autorização do interessado, intervém na gestão...</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl mx-auto w-full my-8">
        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Database className="w-5 h-5 text-emerald-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Single-File Binary</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            All nodes, CSR edges, quantized vectors, and string tables packed into a single <code className="text-neutral-300">.graphite</code> file with CRC32 verification.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Zap className="w-5 h-5 text-teal-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Zero-Copy Mmap</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Direct memory-mapping via <code className="text-neutral-300">memmap2</code> yields 1.6µs context synthesis and 500,000+ QPS with zero heap allocation on read.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Cpu className="w-5 h-5 text-cyan-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">SIMD AVX2 & SQ8</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Vector distance kernels utilize 8 YMM registers and Scalar Int8 quantization for 4x memory savings and optimal CPU throughput.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Network className="w-5 h-5 text-emerald-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Hybrid Search & RRF</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Combines dense semantic vectors with BM25 inverted lexical indexing and legal acronym expansion using Reciprocal Rank Fusion.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Layers className="w-5 h-5 text-teal-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Token-Budgeted MMR</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Enforces strict prompt token limits via Tiktoken BPE counting while maximizing information diversity with zero-alloc Jaccard pruning.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <ShieldCheck className="w-5 h-5 text-cyan-400 mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Model Context Protocol</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Native MCP server integration providing long-term memory tools for Claude Desktop, Cursor, and agentic workflows.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-neutral-900 pt-8 mt-12 text-center text-xs text-neutral-600">
        Graphite is open-source under MIT OR Apache-2.0.
      </footer>
    </div>
  );
}
