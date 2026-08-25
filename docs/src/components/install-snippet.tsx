'use client';

import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

const INSTALL_CMD = 'curl -fsSL https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.sh | bash';

export function InstallSnippet() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL_CMD);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-8">
      <div 
        onClick={handleCopy}
        className="group relative flex items-center justify-between gap-4 px-5 py-3.5 sm:py-4 rounded-xl bg-neutral-100 dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-800 hover:border-neutral-400 dark:hover:border-neutral-700 text-left font-mono text-sm sm:text-base text-neutral-900 dark:text-neutral-100 shadow-sm cursor-pointer transition-all duration-150 active:scale-[0.99]"
      >
        <div className="flex items-center gap-3 min-w-0 overflow-hidden">
          <span className="text-neutral-400 dark:text-neutral-600 select-none font-semibold text-base sm:text-lg shrink-0">
            $
          </span>
          <span className="truncate selection:bg-neutral-300 dark:selection:bg-neutral-700 tracking-tight">
            {INSTALL_CMD}
          </span>
        </div>

        <button
          type="button"
          aria-label="Copy install command"
          className="flex items-center justify-center p-2 rounded-lg text-neutral-500 dark:text-neutral-400 group-hover:text-neutral-950 dark:group-hover:text-white bg-neutral-200/60 dark:bg-neutral-800/80 hover:bg-neutral-300 dark:hover:bg-neutral-700 border border-neutral-300 dark:border-neutral-700/80 transition-colors shrink-0"
        >
          {copied ? (
            <Check className="w-4 h-4 sm:w-5 sm:h-5 text-neutral-950 dark:text-white transition-transform scale-110" />
          ) : (
            <Copy className="w-4 h-4 sm:w-5 sm:h-5 transition-transform" />
          )}
        </button>
      </div>
    </div>
  );
}
