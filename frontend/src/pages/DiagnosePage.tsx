import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { diagnoseImage, listProjects } from "../api/client";
import { DiagnosisResults } from "../components/DiagnosisResults";
import { SavedBanner } from "../components/SavedBanner";
import { SavedPhotoPicker } from "../components/SavedPhotoPicker";
import type { DiagnosisResponse, Project } from "../types/api";

export function DiagnosePage() {
  const [searchParams] = useSearchParams();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState(searchParams.get("project") ?? "");
  const [note, setNote] = useState("");

  useEffect(() => { listProjects().then(setProjects).catch(() => setProjects([])); }, []);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function analyze(reanalyse = false) {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult(await diagnoseImage(file, { projectId: projectId || null, note, reanalyse })); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Analysis failed."); }
    finally { setLoading(false); }
  }

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow">Static image workflow</p>
        <h1>Inspect a leaf image</h1>
        <p>Upload a clear JPEG, PNG, or WebP image. The API reports whether a trained checkpoint or explicit demo classifier produced the result. Every photo and its diagnosis is saved; uploading the same photo again shows the saved diagnosis.</p>
      </section>
      <section className="upload-layout">
        <label className={`dropzone ${preview ? "has-image" : ""}`}>
          {preview ? <img src={preview} alt="Selected leaf preview" /> : <><span className="upload-glyph">+</span><strong>Choose a leaf image</strong><small>JPEG, PNG, or WebP · maximum 10 MB</small></>}
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setResult(null); }} />
        </label>
        <aside className="upload-actions">
          <span className="status-pill">Runtime mode reported with result</span>
          <h2>Analysis pipeline</h2>
          <ol><li>Validate and decode</li><li>Preprocess and isolate leaf</li><li>Configured classification</li><li>Severity status check</li><li>Grounded or demo advisory</li></ol>
          <SavedPhotoPicker disabled={loading} onPick={(picked) => { setFile(picked); setResult(null); }} />
          <div className="upload-fields">
            <label>
              <span>Save this photo to a project (optional)</span>
              <select value={projectId} onChange={(event) => setProjectId(event.target.value)}>
                <option value="">Don't add to a project</option>
                {projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name} · {project.image_count} photo{project.image_count === 1 ? "" : "s"}</option>)}
              </select>
            </label>
            <label>
              <span>Plant label (optional)</span>
              <input value={note} maxLength={300} placeholder="e.g. Plant 3, row 2" onChange={(event) => setNote(event.target.value)} />
            </label>
          </div>
          <button className="button primary" disabled={!file || loading} onClick={() => void analyze()}>{loading ? "Analyzing…" : "Run analysis"}</button>
          {error && <p className="error" role="alert">{error}</p>}
        </aside>
      </section>
      {result && <SavedBanner result={result} busy={loading} onReanalyse={() => void analyze(true)} />}
      {result && <DiagnosisResults result={result} />}
    </main>
  );
}
