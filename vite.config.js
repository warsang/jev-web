import { defineConfig } from "vite";

// Local playground only. COOP/COEP unlock SharedArrayBuffer for the threaded
// onnxruntime-web build; without them ORT falls back to single-threaded WASM.
export default defineConfig({
  root: "dev",
  server: {
    port: 5188,
    strictPort: true,
    headers: {
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    },
  },
  build: { outDir: "../dist-dev", emptyOutDir: true, target: "es2022" },
});
