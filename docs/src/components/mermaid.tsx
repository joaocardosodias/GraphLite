'use client';

import { useEffect, useId, useState } from 'react';
import mermaid from 'mermaid';

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '');
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#000000',
        mainBkg: '#0a0a0a',
        nodeBorder: '#10b981',
        primaryColor: '#0a0a0a',
        primaryTextColor: '#ededed',
        primaryBorderColor: '#10b981',
        lineColor: '#10b981',
        secondaryColor: '#0f172a',
        tertiaryColor: '#1e293b',
        textColor: '#ededed',
        fontSize: '13px',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
      },
    });

    const cleanChart = chart.trim();
    mermaid
      .render(`mermaid-${id}`, cleanChart)
      .then(({ svg }) => {
        setSvg(svg);
        setError(null);
      })
      .catch((err) => {
        console.error('Mermaid render error:', err);
        setError(err.message || 'Failed to render diagram');
      });
  }, [chart, id]);

  if (error) {
    return (
      <div className="p-4 my-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 font-mono text-xs overflow-x-auto">
        <p className="font-semibold mb-1">Diagram Render Error:</p>
        <pre>{chart}</pre>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="flex items-center justify-center p-8 my-4 rounded-xl border border-neutral-800 bg-[#050505] text-neutral-500 text-xs font-mono animate-pulse">
        Rendering diagram...
      </div>
    );
  }

  return (
    <div
      className="my-6 p-6 rounded-xl border border-neutral-800 bg-[#050505] flex justify-center items-center overflow-x-auto shadow-2xl [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
