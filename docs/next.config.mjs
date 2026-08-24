import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

const basePath = process.env.BASE_PATH || '';

/** @type {import('next').NextConfig} */
const config = {
  serverExternalPackages: ['@takumi-rs/core'],
  output: 'export',
  reactStrictMode: true,
  basePath: basePath || undefined,
  images: {
    unoptimized: true,
  },
};

export default withMDX(config);
