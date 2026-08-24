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
    default: 'Graphite — Embedded GraphRAG Database in Rust',
  },
  description:
    'Single-file embedded binary database unifying Knowledge Graphs, SIMD AVX2 Vector Search, BM25 Lexical Search, and Local Cross-Encoder Reranking in pure Rust.',
  metadataBase: new URL('https://joaocardosodias.github.io/Graphite'),
  icons: {
    icon: [
      { url: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/favicon.ico` },
      { url: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/favicon.png`, type: 'image/png' },
    ],
    apple: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/apple-touch-icon.png`,
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen relative bg-black">
        {/* Subtle Faint White Grid Background */}
        <div className="fixed inset-0 pointer-events-none -z-10 bg-grid-white [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_60%,transparent_100%)]" />
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
