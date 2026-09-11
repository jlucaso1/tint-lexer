import { spawnSync, type SpawnSyncOptions } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { brotliCompressSync, constants } from "node:zlib";

const root = process.cwd();

function run(command: string, args: string[], options: SpawnSyncOptions = {}): string {
  const result = spawnSync(command, args, { cwd: root, stdio: "pipe", encoding: "utf8", ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} failed with code ${result.status}:\n${result.stderr}`);
  }
  return result.stdout.trim();
}

async function ensureDir(path: string): Promise<void> {
  await mkdir(path, { recursive: true });
}

export async function buildWasm(): Promise<{ wasmBase64: string; rawBytes: number; brotliBytes: number }> {
  console.log("Compiling Rust tokenizer to WebAssembly...");

  // Try building with nightly build-std + panic=immediate-abort for minimal binary size (~10.4 KiB)
  const isNightlyAvailable = spawnSync("cargo", ["+nightly", "--version"]).status === 0;

  if (isNightlyAvailable) {
    console.log("Using nightly toolchain for a compact wasm build...");
    run(
      "cargo",
      [
        "+nightly",
        "build",
        "-p",
        "tint-tokenizer",
        "--target",
        "wasm32-unknown-unknown",
        "--no-default-features",
        "-Z",
        "build-std=std,panic_abort",
        "--profile",
        "web",
      ],
      {
        env: {
          ...process.env,
          RUSTFLAGS: "-Zunstable-options -Cpanic=immediate-abort -C target-feature=+simd128",
        },
      }
    );
  } else {
    run("cargo", [
      "build",
      "-p",
      "tint-tokenizer",
      "--target",
      "wasm32-unknown-unknown",
      "--no-default-features",
      "--profile",
      "web",
      "--jobs",
      "2",
      "--target-dir",
      "target",
    ]);
  }

  const staging = await mkdtemp(join(root, ".wasm-staging-"));
  try {
    // Raw numeric ABI: zero imports, so wasm-bindgen is skipped entirely.
    // Optimize the module directly.
    console.log("Running wasm-opt...");
    const wasmPath = join(staging, "tint_tokenizer_opt.wasm");
    run("wasm-opt", [
      "target/wasm32-unknown-unknown/web/tint_tokenizer.wasm",
      "-Oz",
      "--all-features",
      "--strip-producers",
      "-o",
      wasmPath,
    ]);

    const bytes = await readFile(wasmPath);
    const imports = WebAssembly.Module.imports(new WebAssembly.Module(bytes));
    if (imports.length !== 0) {
      throw new Error(`Raw tokenizer ABI must have zero imports, found: ${JSON.stringify(imports)}`);
    }
    const wasmBase64 = bytes.toString("base64");
    const brotliBytes = brotliCompressSync(bytes, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
    }).length;

    return { wasmBase64, rawBytes: bytes.length, brotliBytes };
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  const distDir = join(root, "dist");
  await ensureDir(distDir);

  const { wasmBase64, rawBytes, brotliBytes } = await buildWasm();
  console.log(`WASM tokenizer ready: ${rawBytes} bytes raw, ${brotliBytes} bytes Brotli-11.`);

  const distIndexPath = join(distDir, "index.js");
  const srcIndexPath = join(root, "src", "index.ts");
  const webTintPath = join(root, "web", "tint.js");

  const srcCode = (await readFile(srcIndexPath, "utf8")).replace(
    /const WASM_B64 = "[^"]*";/,
    `const WASM_B64 = "${wasmBase64}";`
  );
  await writeFile(srcIndexPath, srcCode, "utf8");

  run("npx", ["esbuild", srcIndexPath, "--outfile=" + distIndexPath, "--format=esm", "--minify"]);
  const compiledJs = await readFile(distIndexPath, "utf8");
  await writeFile(webTintPath, compiledJs, "utf8");

  console.log("Updated dist/index.js, src/index.ts, and web/tint.js with freshly compiled WASM binary.\n");
}

if (process.argv[1]?.endsWith("build.ts")) {
  main().catch((err: Error) => {
    console.error(err);
    process.exit(1);
  });
}
