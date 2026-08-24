import Link from 'next/link';
import { ArrowRight, Terminal, Cpu, Database, Zap, Layers, Network, ShieldCheck } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="relative min-h-[calc(100dvh-4rem)] flex flex-col justify-between max-w-6xl mx-auto px-6 py-12 md:py-16">
      
      {/* Hero Section */}
      <section className="text-center max-w-3xl mx-auto pt-4 md:pt-8">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-white mb-5 leading-tight">
          The Embedded Engine for GraphRAG
        </h1>

        <p className="text-base sm:text-lg text-neutral-400 leading-relaxed mb-8 max-w-2xl mx-auto">
          Graphite combines Knowledge Graphs, SIMD Vector Search, BM25 Indexing, and Cross-Encoder Reranking into a single, zero-dependency <code className="text-sm font-mono text-white bg-neutral-900 px-1.5 py-0.5 rounded border border-neutral-800">.graphite</code> binary file.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/en/docs"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-black bg-white hover:bg-neutral-200 active:scale-[0.98] transition-all shadow-sm"
          >
            Read Documentation
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/en/docs/quickstart"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-neutral-300 bg-neutral-900 border border-neutral-800 hover:bg-neutral-800 hover:text-white active:scale-[0.98] transition-all"
          >
            Quickstart
          </Link>
          <a
            href="https://github.com/joaocardosodias/Graphite"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-neutral-400 hover:text-neutral-200 active:scale-[0.98] transition-all"
          >
            GitHub
          </a>
        </div>
      </section>

      {/* Asymmetric Technical Bento Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl mx-auto w-full my-6">
        
        {/* Large Feature Card (Span 2) */}
        <div className="md:col-span-2 p-6 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-white">
                <Zap className="w-5 h-5" />
                <span className="text-xs font-mono font-semibold uppercase tracking-wider">Zero-Copy Mmap Engine</span>
              </div>
              <span className="text-xs font-mono text-neutral-400 bg-neutral-900 border border-neutral-800 px-2 py-0.5 rounded">
                1.6µs read latency
              </span>
            </div>
            <h3 className="text-base font-semibold text-white mb-2">Memory-Mapped Graph & SIMD Vector Operations</h3>
            <p className="text-xs sm:text-sm text-neutral-400 leading-relaxed">
              Maps binary database pages directly into process virtual memory with <code className="text-neutral-300">memmap2</code>. Vector distance kernels execute over 8 YMM 256-bit AVX2 registers with zero heap allocation on query execution.
            </p>
          </div>
          <div className="mt-6 pt-4 border-t border-neutral-800/60 flex items-center gap-4 text-xs font-mono text-neutral-500">
            <span>500,000+ QPS</span>
            <span>•</span>
            <span>IEEE 802.3 CRC32</span>
            <span>•</span>
            <span>SQ8 Int8 Quantization</span>
          </div>
        </div>

        {/* Compact Card (Span 1) */}
        <div className="p-6 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-white mb-4">
              <Database className="w-5 h-5" />
              <span className="text-xs font-mono font-semibold uppercase tracking-wider">Single File</span>
            </div>
            <h3 className="text-base font-semibold text-white mb-2">Self-Contained Storage</h3>
            <p className="text-xs sm:text-sm text-neutral-400 leading-relaxed">
              Graph topology (CSR), scalar-quantized vectors, inverted BM25 lexical indices, and string tables packed into a single <code className="text-neutral-300">.graphite</code> file.
            </p>
          </div>
          <div className="mt-6 pt-4 border-t border-neutral-800/60 text-xs font-mono text-neutral-500">
            Atomic Safe Rename Writes
          </div>
        </div>

        {/* Medium Card (Span 1) */}
        <div className="p-6 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Network className="w-5 h-5 text-white mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Hybrid Search & RRF</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Combines dense semantic vector projections with BM25 inverted lexical indexing and legal acronym expansion using Reciprocal Rank Fusion.
          </p>
        </div>

        {/* Medium Card (Span 1) */}
        <div className="p-6 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <Layers className="w-5 h-5 text-white mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Token-Budgeted MMR</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Enforces strict prompt token limits via Tiktoken BPE counting while maximizing information diversity with zero-alloc Jaccard pruning.
          </p>
        </div>

        {/* Medium Card (Span 1) */}
        <div className="p-6 rounded-xl border border-neutral-800 bg-[#070707] hover:border-neutral-700 transition-colors">
          <ShieldCheck className="w-5 h-5 text-white mb-3" />
          <h3 className="text-sm font-semibold text-white mb-1.5">Model Context Protocol</h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Native MCP server integration providing long-term memory tools for Claude Desktop, Cursor, and agentic LLM workflows.
          </p>
        </div>

      </section>

      {/* Footer */}
      <footer className="border-t border-neutral-900 pt-8 mt-8 text-center text-xs text-neutral-600">
        Graphite is open-source under MIT OR Apache-2.0.
      </footer>
    </div>
  );
}
