import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  diagnoseImage,
  listProjects,
  loadAgentMemory,
  loadProject,
  loadProjectAgentMemory,
  loadSavedDiagnosis,
  resetAgentMemory,
  runAgent,
} from "../api/client";
import { AgentOutput } from "../components/AgentOutput";
import { photoName, ProjectAgentDashboard } from "../components/ProjectAgentDashboard";
import { SavedBanner } from "../components/SavedBanner";
import { SavedPhotoPicker } from "../components/SavedPhotoPicker";
import type { AgentMemory, AgentName, AgentRunResponse, DiagnosisResponse, GeoLocation, Project, ProjectDetail, StoredImage } from "../types/api";

interface BatchRow { image: StoredImage; response?: AgentRunResponse; error?: string }

interface AgentCard {
  name: AgentName;
  title: string;
  role: string;
  needs: string;
  placeholder: string;
}

const AGENTS: AgentCard[] = [
  { name: "diagnose", title: "Doctor", role: "Compares the image model's top guesses with fact sheets and your answers, and says how sure it is.", needs: "Evidence (runs it first if missing)", placeholder: "Optional: describe what you see, e.g. yellow halo around spots" },
  { name: "retrieve_evidence", title: "Evidence researcher", role: "Searches the knowledge base for fact sheets on each likely disease.", needs: "Nothing", placeholder: "Optional: extra words to search for" },
  { name: "ask_farmer", title: "Question agent", role: "Asks 2–3 simple questions that separate look-alike diseases.", needs: "A diagnosis (runs the Doctor first if missing)", placeholder: "Optional: anything you already know" },
  { name: "assess_risk", title: "Risk & weather", role: "Estimates spread risk from lesion area, likelihood and a 3-day weather forecast.", needs: "A diagnosis; share location for weather", placeholder: "Optional" },
  { name: "recommend_treatment", title: "Treatment", role: "Immediate, field, organic and chemical actions — chemical ones must cite a fact sheet.", needs: "A diagnosis and evidence", placeholder: "Optional: e.g. I prefer organic options" },
  { name: "plan_followup", title: "Follow-up planner", role: "Sets a re-check date, what to look for, and when to call an expert.", needs: "A diagnosis", placeholder: "Optional" },
  { name: "answer_question", title: "Q&A", role: "Answers one specific question about this crop from the fact sheets.", needs: "Your question", placeholder: "Required: e.g. Is it safe to eat fruit from this plant?" },
];

const SESSION_KEY = "cropai.agents.session";

/** Whether the agent's result is already saved for a photo (so running it again costs nothing). */
function hasSaved(agent: AgentName, memory: AgentMemory | null | undefined): boolean {
  if (!memory) return false;
  switch (agent) {
    case "retrieve_evidence": return memory.passages > 0;
    case "diagnose": return memory.has_diagnosis;
    case "ask_farmer": return memory.pending_questions.length > 0;
    case "assess_risk": return memory.has_risk;
    case "recommend_treatment": return memory.has_treatment;
    case "plan_followup": return memory.has_follow_up;
    case "answer_question": return false;
  }
}
const STATUS_LABEL = { ok: "done", fallback: "rule-based", skipped: "skipped", error: "failed" } as const;

function remembered(): string | null {
  try { return localStorage.getItem(SESSION_KEY); } catch { return null; }
}

function remember(sessionId: string | null) {
  try {
    if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
    else localStorage.removeItem(SESSION_KEY);
  } catch { /* storage unavailable: the page still works for this visit */ }
}

export function AgentsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(remembered);
  const [memory, setMemory] = useState<AgentMemory | null>(null);
  const [selected, setSelected] = useState<AgentCard>(AGENTS[0]);
  const [message, setMessage] = useState("");
  const [location, setLocation] = useState<GeoLocation | null>(null);
  const [locationNote, setLocationNote] = useState("");
  const [result, setResult] = useState<AgentRunResponse | null>(null);
  const [busy, setBusy] = useState<"photo" | "agent" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const resultRef = useRef<HTMLElement>(null);
  const [parameters, setParameters] = useSearchParams();
  const projectId = parameters.get("project") ?? "";
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [memories, setMemories] = useState<Record<string, AgentMemory | null>>({});
  const [batch, setBatch] = useState<BatchRow[] | null>(null);
  const [batchProgress, setBatchProgress] = useState<string | null>(null);
  const [mode, setMode] = useState<"photo" | "project">(projectId ? "project" : "photo");
  const [rerunAll, setRerunAll] = useState(false);

  useEffect(() => { listProjects().then(setProjects).catch(() => setProjects([])); }, []);

  const loadDashboard = useCallback(async () => {
    if (!projectId) { setProject(null); setMemories({}); return; }
    try {
      const [detail, rows] = await Promise.all([loadProject(projectId), loadProjectAgentMemory(projectId)]);
      setProject(detail);
      setMemories(Object.fromEntries(rows.map((row) => [row.diagnosis_id, row.memory])));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not load the project.");
    }
  }, [projectId]);

  useEffect(() => { setBatch(null); void loadDashboard(); }, [loadDashboard]);

  function chooseProject(value: string) {
    setResult(null); setBatch(null); setError(null);
    setParameters(value ? { project: value } : {});
  }

  function chooseMode(next: "photo" | "project") {
    setMode(next);
    if (next === "photo") chooseProject("");
  }

  async function selectPhoto(image: StoredImage) {
    setResult(null); setError(null); setFile(null);
    setSessionId(image.diagnosis_id);
    remember(image.diagnosis_id);
    setDiagnosis(await loadSavedDiagnosis(image.diagnosis_id).catch(() => null));
  }

  async function runOnAll() {
    if (!project || project.images.length === 0) return;
    setBusy("agent"); setError(null); setResult(null);
    const rows: BatchRow[] = [];
    for (const [index, image] of project.images.entries()) {
      setBatchProgress(`Running ${selected.title} on ${index + 1} of ${project.images.length}: ${photoName(image)}`);
      try {
        rows.push({ image, response: await runAgent(selected.name, image.diagnosis_id, message.trim(), location, rerunAll) });
      } catch (problem) {
        rows.push({ image, error: problem instanceof Error ? problem.message : "failed" });
      }
      setBatch([...rows]);
    }
    setBatchProgress(null);
    setBusy(null);
    await loadDashboard();
    if (sessionId) loadAgentMemory(sessionId).then(setMemory).catch(() => undefined);
    requestAnimationFrame(() => document.getElementById("batch-results")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  useEffect(() => {
    if (!sessionId) { setMemory(null); return; }
    loadAgentMemory(sessionId).then(setMemory).catch(() => { setSessionId(null); remember(null); });
  }, [sessionId]);

  async function analyze(reanalyse = false) {
    if (!file) return;
    setBusy("photo"); setError(null); setResult(null);
    try {
      const response = await diagnoseImage(file, { reanalyse });
      setDiagnosis(response);
      setSessionId(response.session_id);
      remember(response.session_id);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Photo analysis failed.");
    } finally { setBusy(null); }
  }

  async function run(rerun = false) {
    if (!sessionId) return;
    setBusy("agent"); setError(null); setResult(null);
    try {
      const response = await runAgent(selected.name, sessionId, message.trim(), location, rerun);
      setResult(response);
      if (projectId) void loadDashboard();
      // The result is drawn below the agent cards, often off-screen: bring it into view.
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      setMemory(response.memory);
      setMessage("");
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "The agent failed.");
    } finally { setBusy(null); }
  }

  async function reset() {
    if (!sessionId) return;
    await resetAgentMemory(sessionId).catch(() => undefined);
    setResult(null);
    setMemory(await loadAgentMemory(sessionId).catch(() => null));
  }

  function shareLocation() {
    if (!navigator.geolocation) { setLocationNote("Location is not available in this browser."); return; }
    setLocationNote("Getting location…");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          latitude: Math.round(position.coords.latitude * 100) / 100,
          longitude: Math.round(position.coords.longitude * 100) / 100,
        });
        setLocationNote("Location shared (rounded to ~1 km).");
      },
      () => setLocationNote("Location permission was denied; weather will be skipped."),
      { timeout: 10000 },
    );
  }

  const needsMessage = selected.name === "answer_question" && !message.trim();
  const selectedPhoto = project?.images.find((image) => image.diagnosis_id === sessionId) ?? null;
  const savedHere = Boolean(sessionId) && (!project || selectedPhoto !== null) && !message.trim() && hasSaved(selected.name, memory);
  const savedCount = project ? project.images.filter((image) => hasSaved(selected.name, memories[image.diagnosis_id])).length : 0;
  const prediction = diagnosis?.prediction;

  return (
    <main className="page-shell agents-page">
      <section className="page-intro">
        <p className="eyebrow">Agent workbench</p>
        <h1>Run each agent on its own</h1>
        <p>Analyse a leaf photo once, then call any agent from the team directly. Results are remembered for this photo, so agents build on each other — the Doctor's diagnosis is used by Treatment, Risk and Follow-up.</p>
      </section>

      <section className="panel agents-project-bar" aria-label="What the agents work on">
        <div className="choice-group" role="radiogroup" aria-label="Run the agents on">
          <span className="choice-label">Run the agents on</span>
          <button type="button" role="radio" aria-checked={mode === "photo"} className={`choice ${mode === "photo" ? "selected" : ""}`} onClick={() => chooseMode("photo")}>
            <strong>One photo</strong><small>Upload a leaf or pick a saved photo</small>
          </button>
          <button type="button" role="radio" aria-checked={mode === "project"} className={`choice ${mode === "project" ? "selected" : ""}`} onClick={() => chooseMode("project")}>
            <strong>A project</strong><small>Every plant in one of your projects</small>
          </button>
        </div>
        {mode === "project" && (
          <label className="field">
            <span>Project</span>
            <select value={projectId} onChange={(event) => chooseProject(event.target.value)}>
              <option value="" disabled>Choose a project…</option>
              {projects.map((item) => <option key={item.project_id} value={item.project_id}>{item.name} · {item.image_count} photo{item.image_count === 1 ? "" : "s"}</option>)}
            </select>
            {projects.length === 0 && <small className="muted">No projects yet — create one on the <Link to="/projects">Projects page</Link>.</small>}
          </label>
        )}
      </section>

      {project && <ProjectAgentDashboard project={project} memories={memories} selectedId={sessionId} onSelect={(image) => void selectPhoto(image)} />}

      {mode === "photo" && <section className="panel agents-photo">
        <label className="agents-file">
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          <span>{file ? file.name : "Choose a leaf photo"}</span>
        </label>
        <SavedPhotoPicker disabled={busy !== null} onPick={setFile} />
        <button className="button primary" disabled={!file || busy !== null} onClick={() => void analyze()}>
          {busy === "photo" ? "Analysing…" : "Analyse photo"}
        </button>
        <div className="agents-photo-status">
          {prediction ? (
            <>
              <strong>{prediction.crop} – {prediction.disease}</strong>
              <span>{Math.round(prediction.confidence * 100)}% image-model confidence</span>
              {diagnosis?.is_mock && <span className="tag caution-tag">DEMO MODE — placeholder prediction</span>}
            </>
          ) : sessionId ? (
            <span className="muted">Using your previous photo. Upload a new one to start over.</span>
          ) : (
            <span className="muted">Agents need a diagnosed photo first.</span>
          )}
        </div>
      </section>}

      {selectedPhoto && (
        <p className="agents-selected muted">Working on: <strong>{photoName(selectedPhoto)}</strong> — {selectedPhoto.crop && selectedPhoto.disease ? `${selectedPhoto.crop} – ${selectedPhoto.disease}` : "no confident prediction"}</p>
      )}
      {project && !selectedPhoto && project.images.length > 0 && (
        <p className="agents-selected muted">Click a photo above to run one agent on it, or run an agent on every photo at once.</p>
      )}
      {!project && diagnosis && <SavedBanner result={diagnosis} busy={busy === "photo"} onReanalyse={() => void analyze(true)} />}

      {memory && (
        <section className="agents-memory" aria-label="What the team already knows">
          <span>Team memory:</span>
          <span className={`tag ${memory.passages ? "risk-low" : ""}`}>{memory.passages} fact-sheet passages</span>
          {([["has_diagnosis", "Diagnosis"], ["has_risk", "Risk"], ["has_treatment", "Treatment"], ["has_follow_up", "Follow-up"]] as const).map(([key, label]) => (
            <span key={key} className={`tag ${memory[key] ? "risk-low" : ""}`}>{memory[key] ? "✓" : "–"} {label}</span>
          ))}
          {memory.answered_questions > 0 && <span className="tag">{memory.answered_questions} answer(s) given</span>}
          <button type="button" className="link-button" onClick={reset}>Forget and start over</button>
        </section>
      )}

      <section className="agents-layout">
        <div className="agents-grid" role="listbox" aria-label="Choose an agent">
          {AGENTS.map((agent) => (
            <button
              key={agent.name}
              type="button"
              role="option"
              aria-selected={selected.name === agent.name}
              className={`agent-card ${selected.name === agent.name ? "selected" : ""}`}
              onClick={() => setSelected(agent)}
            >
              <strong>{agent.title}{hasSaved(agent.name, memory) && <span className="tag risk-low saved-tag">✓ saved</span>}</strong>
              <p>{agent.role}</p>
              <small>Needs: {agent.needs}</small>
            </button>
          ))}
        </div>

        <div className="panel agents-run">
          <h2>{selected.title}</h2>
          {memory && memory.pending_questions.length > 0 && selected.name !== "ask_farmer" && (
            <div className="callout">
              <strong>The question agent asked:</strong>
              <ol>{memory.pending_questions.map((question) => <li key={question}>{question}</li>)}</ol>
              <small>Your message below is taken as the answer, and the Doctor re-diagnoses.</small>
            </div>
          )}
          <textarea
            value={message}
            maxLength={2000}
            placeholder={selected.placeholder}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                if (sessionId && !busy && !needsMessage) void run();
              }
            }}
          />
          <div className="agents-run-actions">
            {selected.name === "assess_risk" && (
              <button type="button" className="button secondary" onClick={shareLocation} disabled={location !== null}>
                {location ? "Location shared" : "Share location for weather"}
              </button>
            )}
            {project && project.images.length > 0 && (
              <button className="button secondary" disabled={busy !== null || needsMessage} onClick={() => void runOnAll()}>
                {selected.title} for all {project.images.length} photos
              </button>
            )}
            <button className="button primary" disabled={!sessionId || busy !== null || needsMessage || (project !== null && !selectedPhoto)} onClick={() => void run()}>
              {busy === "agent" ? `Running ${selected.title}…` : savedHere ? `Show saved ${selected.title} result` : `Run ${selected.title}`}
            </button>
          </div>
          {savedHere && !busy && (
            <small className="muted">
              Already worked out for this photo — showing it costs no new analysis.{" "}
              <button type="button" className="link-button" onClick={() => void run(true)}>Run {selected.title} again</button>
            </small>
          )}
          {project && project.images.length > 0 && (
            <label className="project-check">
              <input type="checkbox" checked={rerunAll} onChange={(event) => setRerunAll(event.target.checked)} />
              For all photos: run again even where a result is saved ({savedCount} of {project.images.length} saved)
            </label>
          )}
          {locationNote && selected.name === "assess_risk" && <small className="muted">{locationNote}</small>}
          {busy === "agent" && <small className="muted">{batchProgress ?? "The agents are working; this can take 5–30 seconds."}</small>}
          {project && <small className="muted">"Run on all" makes about 1–3 language-model calls per photo.</small>}
          {result && !busy && (
            <button type="button" className="link-button" onClick={() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>
              {AGENTS.find((agent) => agent.name === result.agent)?.title} result is ready ↓
            </button>
          )}
          {error && <p className="chat-error" role="alert">{error}</p>}
        </div>
      </section>

      {batch && (
        <section id="batch-results" className="panel agents-result" aria-live="polite">
          <div className="section-heading">
            <h2>{AGENTS.find((agent) => agent.name === batch[0]?.response?.agent)?.title ?? selected.title} on every photo</h2>
            <span className="muted">{batch.length} of {project?.images.length ?? batch.length} done</span>
          </div>
          {batch.map((row) => (
            <details key={row.image.diagnosis_id} className="batch-row" open={batch.length <= 3}>
              <summary>
                <strong>{photoName(row.image)}</strong>
                <span className="muted"> — {row.error ? `failed: ${row.error}` : row.response?.steps.at(-1)?.summary ?? "done"}</span>
                {row.response?.reused && <span className="tag risk-low saved-tag">saved</span>}
              </summary>
              {row.response && (
                <>
                  {row.response.is_mock && <p className="tag caution-tag">DEMO MODE — rule-based agents, no language model</p>}
                  <AgentOutput agent={row.response.agent} data={row.response.output} />
                </>
              )}
            </details>
          ))}
        </section>
      )}

      {result && (
        <section ref={resultRef} className="panel agents-result" aria-live="polite">
          <div className="section-heading">
            <h2>{AGENTS.find((agent) => agent.name === result.agent)?.title} result</h2>
            <span className="muted">
              {result.llm_calls} LLM call{result.llm_calls === 1 ? "" : "s"} · {(result.total_ms / 1000).toFixed(1)} s
            </span>
          </div>
          {result.is_mock && <p className="tag caution-tag">DEMO MODE — rule-based agents, no language model</p>}
          {result.reused && <p className="saved-banner">Saved result for this photo — nothing was re-run. <button type="button" className="link-button" disabled={busy !== null} onClick={() => void run(true)}>Run again</button></p>}
          <AgentOutput agent={result.agent} data={result.output} />

          {result.warnings.length > 0 && (
            <ul className="pipeline-warnings">{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
          )}

          <details className="agent-trace" open={result.steps.length > 1}>
            <summary>Steps run ({result.steps.length})</summary>
            <ol>
              {result.steps.map((step, index) => (
                <li key={`${step.action}-${index}`} className={`agent-step status-${step.status}`}>
                  <strong>{step.agent}</strong>
                  <span className="agent-step-status">{STATUS_LABEL[step.status]}</span>
                  <p>{step.summary}</p>
                  <small>{step.decided_by === "user" ? "You asked for this agent" : `Ran automatically — ${step.reason}`}{step.ms ? ` · ${Math.round(step.ms)} ms` : ""}</small>
                </li>
              ))}
            </ol>
          </details>

          {result.sources.length > 0 && (
            <div className="agent-output-block">
              <h4>Sources</h4>
              {result.sources.map((source) => (
                <div className="source-card" key={source.source_id}>
                  <span>{source.source_id}</span>
                  <h3>{source.title}</h3>
                  <p>{source.excerpt.slice(0, 220)}…</p>
                  {source.url && <a href={source.url} target="_blank" rel="noreferrer">Open source</a>}
                </div>
              ))}
            </div>
          )}

          <details className="agent-raw">
            <summary>Raw JSON (for the research report)</summary>
            <pre>{JSON.stringify(result.output, null, 2)}</pre>
          </details>
        </section>
      )}
    </main>
  );
}
