import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { GraphiteLogo } from '@/components/logo';
import { gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    i18n: true,
    nav: {
      title: (
        <div className="flex items-center gap-2.5 font-bold tracking-tight">
          <GraphiteLogo className="w-5 h-5 text-neutral-950 dark:text-white transition-colors" />
          <span className="text-base text-neutral-950 dark:text-white transition-colors">Graphite</span>
          <span className="hidden sm:inline-block text-[11px] font-mono font-medium px-1.5 py-0.5 rounded bg-neutral-100 dark:bg-neutral-900 text-neutral-700 dark:text-neutral-300 border border-neutral-200 dark:border-neutral-800 transition-colors">
            v0.1
          </span>
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
