'use client';
import SearchDialog from '@/components/search';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { type ReactNode } from 'react';
import { usePathname, useRouter } from 'next/navigation';

const locales = [
  { locale: 'en', name: 'English' },
  { locale: 'pt', name: 'Português' },
];

export function Provider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const segments = pathname.split('/').filter(Boolean);
  const currentLocale = segments[0] === 'pt' ? 'pt' : 'en';

  const onLocaleChange = (newLocale: string) => {
    const parts = pathname.split('/').filter(Boolean);
    if (parts.length > 0 && (parts[0] === 'en' || parts[0] === 'pt')) {
      parts[0] = newLocale;
    } else {
      parts.unshift(newLocale);
    }
    router.push(`/${parts.join('/')}`);
  };

  return (
    <RootProvider
      search={{ SearchDialog }}
      i18n={{
        locale: currentLocale,
        locales,
        onLocaleChange,
      }}
    >
      {children}
    </RootProvider>
  );
}
