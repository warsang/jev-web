import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Static promo/demo site for the npm README. Deployed to GitHub Pages by
// .github/workflows/pages.yml, so `base` has to match the project path.
const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: "demo",
  base: process.env.DEMO_BASE ?? "/jev-web/",
  build: {
    // Absolute, not "../dist-demo": a relative outDir depends on `root`, and
    // silently writes outside the repo when a config's root differs.
    outDir: path.resolve(here, "dist-demo"),
    emptyOutDir: true,
    target: "es2022",
    // transformers.js resolves the ONNX runtime wasm at runtime (jsdelivr by
    // default). Pre-bundling onnxruntime-web makes Vite try to inline a 26 MB
    // asyncify binary, which breaks the build for no benefit.
    chunkSizeWarningLimit: 4096,
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
