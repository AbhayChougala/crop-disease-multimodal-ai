import type { DiagnosisResponse } from "../types/api";

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
export const WS_URL =
  import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/api/v1/realtime/ws";

async function parseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

export async function diagnoseImage(file: File | Blob): Promise<DiagnosisResponse> {
  const form = new FormData();
  form.append("file", file, file instanceof File ? file.name : "camera.jpg");
  const response = await fetch(`${API_BASE}/diagnose`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as DiagnosisResponse;
}

export async function sendChatMessage(sessionId: string, message: string) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as {
    answer: string;
    sources: Array<{ source_id: string; title: string; url?: string }>;
    warnings: string[];
    is_mock: boolean;
  };
}
