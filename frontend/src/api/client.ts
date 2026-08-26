/** HTTP client helpers for the research backend. */

/** Same-origin proxy avoids browser CORS and direct-port drift in local development. */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export async function errorMessage(response: Response, fallback: string): Promise<string> {
  const payload: unknown = await response.json().catch(() => undefined);
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => typeof item === "object" && item !== null && "msg" in item ? String((item as { msg: unknown }).msg) : "")
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
  }
  return fallback;
}

export async function requestJson<T>(path: string, fallback: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) throw new Error(await errorMessage(response, fallback));
  return response.json() as Promise<T>;
}
