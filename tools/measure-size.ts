import { readFile } from "node:fs/promises";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";
import { resolve } from "node:path";
import { transform } from "esbuild";

interface SizeEntry {
  file: string;
  raw: number;
  minified: number;
  gzip: number;
  brotli: number;
}

const targetFiles = [
  "dist/index.js",
  "dist/index.d.ts",
  "dist/worker.js",
  "dist/worker.d.ts",
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(2)} KiB`;
}

export async function measureBundle(root: string = process.cwd()): Promise<SizeEntry[]> {
  const results: SizeEntry[] = [];

  for (const relativePath of targetFiles) {
    const fullPath = resolve(root, relativePath);
    try {
      const buffer = await readFile(fullPath);
      const text = buffer.toString("utf8");

      let minifiedBytes = buffer.length;
      if (relativePath.endsWith(".js")) {
        const minified = await transform(text, { minify: true, target: "es2022" });
        minifiedBytes = Buffer.byteLength(minified.code);
      }

      const gzipBytes = gzipSync(buffer).length;
      const brotliBytes = brotliCompressSync(buffer, {
        params: {
          [constants.BROTLI_PARAM_QUALITY]: 11,
        },
      }).length;

      results.push({
        file: relativePath,
        raw: buffer.length,
        minified: minifiedBytes,
        gzip: gzipBytes,
        brotli: brotliBytes,
      });
    } catch (error) {
      const err = error as NodeJS.ErrnoException;
      if (err.code !== "ENOENT") throw error;
    }
  }

  return results;
}

async function main(): Promise<void> {
  const entries = await measureBundle();

  console.log("\ntint-lexer bundle sizes (Brotli-11 / Gzip):\n");
  console.log("| File | Raw Size | Minified | Gzip | Brotli-11 |");
  console.log("| :--- | :---: | :---: | :---: | :---: |");

  let totalRaw = 0;
  let totalMin = 0;
  let totalGzip = 0;
  let totalBrotli = 0;

  for (const entry of entries) {
    totalRaw += entry.raw;
    totalMin += entry.minified;
    totalGzip += entry.gzip;
    totalBrotli += entry.brotli;
    console.log(
      `| \`${entry.file}\` | ${formatBytes(entry.raw)} | ${formatBytes(entry.minified)} | ${formatBytes(entry.gzip)} | **${formatBytes(entry.brotli)}** |`
    );
  }

  console.log(
    `| **Total** | **${formatBytes(totalRaw)}** | **${formatBytes(totalMin)}** | **${formatBytes(totalGzip)}** | **${formatBytes(totalBrotli)}** |\n`
  );
}

if (process.argv[1]?.endsWith("measure-size.ts")) {
  main().catch((err: Error) => {
    console.error(err);
    process.exit(1);
  });
}
