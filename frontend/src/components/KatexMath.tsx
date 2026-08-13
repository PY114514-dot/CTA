import { useMemo } from "react";
import type { ReactNode } from "react";
import katex, { ParseError } from "katex";

export interface KatexBlockProps {
  /** LaTeX source string. */
  math: string;
  /** When provided, parse errors are delegated to this renderer instead of KaTeX's inline red text. */
  renderError?: (error: Error) => ReactNode;
}

/**
 * Display-mode KaTeX block — lightweight replacement for react-katex's `BlockMath`.
 *
 * react-katex pulls in its own nested katex copy, whose lexer regex contains lone
 * surrogate escapes (\uD800-\uDFFF). Vite 8's Rolldown-based dependency pre-bundling
 * rewrites those escapes into literal surrogates, which cannot be encoded in UTF-8
 * and degrade to U+FFFD, mangling the token regex so every control sequence fails
 * with "Undefined control sequence: \o". We use the top-level katex directly;
 * vite.config.ts excludes it from pre-bundling and rewrites the surrogate escapes
 * so both dev serving and production builds stay intact.
 */
export function KatexBlock({ math, renderError }: KatexBlockProps) {
  const rendered = useMemo(() => {
    try {
      return {
        html: katex.renderToString(math, {
          displayMode: true,
          throwOnError: !!renderError,
        }),
        error: null as Error | null,
      };
    } catch (error) {
      if (error instanceof ParseError || error instanceof TypeError) {
        return { html: null as string | null, error: error as Error };
      }
      throw error;
    }
  }, [math, renderError]);

  if (rendered.error) {
    return <>{renderError ? renderError(rendered.error) : rendered.error.message}</>;
  }
  return <span className="katex-block" dangerouslySetInnerHTML={{ __html: rendered.html ?? "" }} />;
}
