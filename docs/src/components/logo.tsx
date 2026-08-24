import React from 'react';

export function GraphiteLogo({ className = 'w-6 h-6' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 534 591"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Graphite Logo"
    >
      <polygon
        points="267,64 472,178 472,414 267,529 62,414 62,178"
        stroke="currentColor"
        strokeWidth="28"
        strokeLinejoin="round"
        fill="none"
      />
      <circle cx="267" cy="64" r="43.5" fill="currentColor" />
      <circle cx="472" cy="178" r="43.5" fill="currentColor" />
      <circle cx="472" cy="414" r="43.5" fill="currentColor" />
      <circle cx="267" cy="529" r="43.5" fill="currentColor" />
      <circle cx="62" cy="414" r="43.5" fill="currentColor" />
      <circle cx="62" cy="178" r="43.5" fill="currentColor" />
    </svg>
  );
}
