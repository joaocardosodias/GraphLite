import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Provider } from '@/components/provider';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    template: '%s | Graphite',
    default: 'Graphite — Embedded GraphRAG & AI Agent Memory Database in Rust',
  },
  description:
    'Single-file embedded binary database unifying Knowledge Graphs, SIMD AVX2 Vector Search, BM25 Lexical Search, and Local Cross-Encoder Reranking in pure Rust.',
  metadataBase: new URL('https://joaocardosodias.github.io/Graphite'),
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
