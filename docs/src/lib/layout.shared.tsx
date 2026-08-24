import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { GraphiteLogo } from '@/components/logo';
import { gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    i18n: true,
    nav: {
      title: (
        <div className="flex items-center gap-2.5 font-bold tracking-tight">
          <GraphiteLogo className="w-5 h-5 text-white" />
          <span className="text-base text-white">Graphite</span>
          <span className="hidden sm:inline-block text-[11px] font-mono font-medium px-1.5 py-0.5 rounded bg-neutral-900 text-neutral-300 border border-neutral-800">
            v0.1
          </span>
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
