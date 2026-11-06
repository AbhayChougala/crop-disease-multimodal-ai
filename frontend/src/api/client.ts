import type {
  AgentMemory,
  AgentName,
  AgentRunResponse,
  ChatResponse,
  ChatSession,
  DiagnosisResponse,
  GeoLocation,
  Project,
  ProjectDetail,
  ProjectPhotoMemory,
  StoredImage,
  StoredImageList,
} from "../types/api";

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

export interface DiagnoseOptions {
  /** File the photo under this project. */
  projectId?: string | null;
  /** Label for the photo, e.g. "Plant 3, row 2". */
  note?: string;
  /** Run the model again even if this exact photo already has a saved diagnosis. */
  reanalyse?: boolean;
}

export async function diagnoseImage(file: File | Blob, options: DiagnoseOptions = {}): Promise<DiagnosisResponse> {
  const form = new FormData();
  form.append("file", file, file instanceof File ? file.name : "camera.jpg");
  if (options.projectId) form.append("project_id", options.projectId);
  if (options.note?.trim()) form.append("note", options.note.trim());
  const query = options.reanalyse ? "?reanalyse=true" : "";
  const response = await fetch(`${API_BASE}/diagnose${query}`, { method: "POST", body: form });
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
  return (await response.json()) as ChatResponse;
}

export async function createChatSession(
  mode: "general" | "diagnosis",
  diagnosisId?: string,
  preferredLanguage: "en" | "kn" | "hi" = "en",
  projectId?: string,
): Promise<ChatSession> {
  const response = await fetch(`${API_BASE}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      diagnosis_id: diagnosisId,
      project_id: projectId,
      preferred_language: preferredLanguage,
    }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as ChatSession;
}

export async function loadChatSession(sessionId: string): Promise<ChatSession> {
  const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as ChatSession;
}

export async function sendSessionMessage(
  sessionId: string,
  message: string,
  preferredLanguage: "en" | "kn" | "hi",
  location?: GeoLocation | null,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, preferred_language: preferredLanguage, location: location ?? undefined }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as ChatResponse;
}

/** Link a new photo's diagnosis to a diagnosis chat; the agents compare old and new. */
export async function reassessSession(
  sessionId: string,
  newDiagnosisId: string,
  location?: GeoLocation | null,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}/reassess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ diagnosis_id: newDiagnosisId, location: location ?? undefined }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as ChatResponse;
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await parseError(response));
}

/** Run one agent on its own for a diagnosis (missing prerequisites run first). */
export async function runAgent(
  agent: AgentName,
  sessionId: string,
  message: string,
  location?: GeoLocation | null,
  rerun = false,
): Promise<AgentRunResponse> {
  const response = await fetch(`${API_BASE}/agents/${agent}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, location: location ?? undefined, rerun }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as AgentRunResponse;
}

export async function loadAgentMemory(sessionId: string): Promise<AgentMemory> {
  const response = await fetch(`${API_BASE}/agents/memory/${sessionId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as AgentMemory;
}

export async function resetAgentMemory(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/agents/memory/${sessionId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await parseError(response));
}

/** Leaf photos saved in the backend's local database (History page). */
export async function listStoredImages(limit = 200): Promise<StoredImageList> {
  const response = await fetch(`${API_BASE}/images?limit=${limit}`);
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as StoredImageList;
}

export function storedImageUrl(diagnosisId: string): string {
  return `${API_BASE}/images/${encodeURIComponent(diagnosisId)}/file`;
}

export async function deleteStoredImage(diagnosisId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/images/${encodeURIComponent(diagnosisId)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await parseError(response));
}

/** Download a stored photo as a File, so it can be analysed again like a fresh upload. */
export async function loadStoredImageFile(image: StoredImage): Promise<File> {
  const response = await fetch(storedImageUrl(image.diagnosis_id));
  if (!response.ok) throw new Error(await parseError(response));
  const blob = await response.blob();
  return new File([blob], image.filename, { type: image.content_type });
}

/** The full saved diagnosis of a stored photo (same shape as POST /diagnose). */
export async function loadSavedDiagnosis(diagnosisId: string): Promise<DiagnosisResponse> {
  const response = await fetch(`${API_BASE}/reports/diagnosis/${encodeURIComponent(diagnosisId)}.json`);
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as DiagnosisResponse;
}

async function sendJson<T>(path: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return (response.status === 204 ? undefined : await response.json()) as T;
}

/** Change a stored photo's label and/or project (projectId null removes it from its project). */
export function updateStoredImage(diagnosisId: string, changes: { note?: string; project_id?: string | null }) {
  return sendJson<StoredImage>(`/images/${encodeURIComponent(diagnosisId)}`, "PATCH", changes);
}

export const listProjects = () => sendJson<Project[]>("/projects", "GET");
export const createProject = (name: string, description = "") =>
  sendJson<Project>("/projects", "POST", { name, description });
export const loadProject = (projectId: string) =>
  sendJson<ProjectDetail>(`/projects/${encodeURIComponent(projectId)}`, "GET");
export const updateProject = (projectId: string, changes: { name?: string; description?: string }) =>
  sendJson<Project>(`/projects/${encodeURIComponent(projectId)}`, "PATCH", changes);
export const deleteProject = (projectId: string) =>
  sendJson<void>(`/projects/${encodeURIComponent(projectId)}`, "DELETE");
export const loadProjectAgentMemory = (projectId: string) =>
  sendJson<ProjectPhotoMemory[]>(`/agents/projects/${encodeURIComponent(projectId)}/memory`, "GET");
