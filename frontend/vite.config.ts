import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

/**
 * KaTeX's lexer builds its token regex from string literals containing lone
 * surrogate escapes (\uD800-\uDFFF). When Rolldown rewrites the module (dep
 * pre-bundling / production build) it emits those code units as UTF-8, which
 * cannot represent lone surrogates — they degrade to U+FFFD and corrupt the
 * token regex, making every control sequence fail with
 * "Undefined control sequence: \o". Rewriting the escapes to their runtime
 * form (\\uD800) keeps the string values pure ASCII; `new RegExp()` interprets
 * them identically, so behavior is unchanged.
 */
function katexSurrogateEscapePlugin(): Plugin {
  return {
    name: "patch-katex-surrogate-escapes",
    enforce: "pre",
    transform(code, id) {
      if (!id.includes("katex") || !/\\u(?:D800|DBFF|DC00|DFFF)/.test(code)) {
        return null;
      }
      return code.replace(/\\u(D800|DBFF|DC00|DFFF)/g, "\\\\$&");
    },
  };
}

export default defineConfig({
  plugins: [react(), katexSurrogateEscapePlugin()],
  server: {
    host: "127.0.0.1",
    port: 5275,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8103",
        changeOrigin: true,
      },
    },
  },
  optimizeDeps: {
    // Serve katex's own ESM dist as-is (its \uD800 escapes stay intact in the
    // source text); a pre-bundled rewrite would corrupt them (see plugin above).
    exclude: ["katex"],
  },
  build: {
    chunkSizeWarningLimit: 3000,
  },
});
