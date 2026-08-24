'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Page() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/en/docs');
  }, [router]);

  return (
    <div className="flex items-center justify-center min-h-[50vh] text-xs font-mono text-neutral-500">
      Redirecting to /en/docs...
    </div>
  );
}
