import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { deleteProject, diagnoseImage, loadProject, updateProject, updateStoredImage } from "../api/client";
import { PhotoCard } from "../components/PhotoCard";
import { SavedPhotoPicker } from "../components/SavedPhotoPicker";
import type { ProjectDetail } from "../types/api";

const labelFromFilename = (name: string) => name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();

/** One project: upload several leaves/plants, see each diagnosis and a per-disease summary. */
export function ProjectPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [useFilenames, setUseFilenames] = useState(true);
  const [progress, setProgress] = useState<string | null>(null);
  const [name, setName] = useState("");

  const load = useCallback(async () => {
    try {
      const detail = await loadProject(projectId);
      setProject(detail);
      setName(detail.name);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not load the project.");
    }
  }, [projectId]);

  useEffect(() => { void load(); }, [load]);

  async function upload(selected: File[], labelWithFilename = useFilenames) {
    if (selected.length === 0) return;
    setError(null);
    const failures: string[] = [];
    let reused = 0;
    for (const [index, file] of selected.entries()) {
      setProgress(`Analysing ${index + 1} of ${selected.length}: ${file.name}`);
      try {
        const result = await diagnoseImage(file, { projectId, note: labelWithFilename ? labelFromFilename(file.name) : "" });
        if (result.from_saved) reused += 1;
      } catch (problem) {
        failures.push(`${file.name}: ${problem instanceof Error ? problem.message : "failed"}`);
      }
    }
    setProgress(reused ? `${reused} photo${reused === 1 ? " was" : "s were"} already analysed; the saved diagnosis was reused.` : null);
    if (failures.length) setError(failures.join(" · "));
    setFiles([]);
    await load();
  }

  async function rename() {
    if (!project || !name.trim() || name.trim() === project.name) return;
    try { await updateProject(projectId, { name: name.trim() }); await load(); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Rename failed."); }
  }

  async function removeFromProject(diagnosisId: string) {
    try { await updateStoredImage(diagnosisId, { project_id: null }); await load(); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Could not remove the photo."); }
  }

  async function removeProject() {
    if (!window.confirm("Delete this project? Its photos and diagnoses stay in History.")) return;
    try { await deleteProject(projectId); navigate("/projects"); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Delete failed."); }
  }

  if (!project) {
    return (
      <main className="page-shell">
        <p className="eyebrow"><Link to="/projects">← Projects</Link></p>
        {error ? <p className="chat-error" role="alert">{error}</p> : <p className="muted">Loading…</p>}
      </main>
    );
  }

  const busy = progress !== null && progress.startsWith("Analysing");

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow"><Link to="/projects">← Projects</Link></p>
        <input className="project-title" value={name} maxLength={120} aria-label="Project name" onChange={(event) => setName(event.target.value)} onBlur={() => void rename()} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} />
        {project.description && <p>{project.description}</p>}
        <div className="button-row project-links">
          <Link className="button primary" to={`/ask?project=${project.project_id}`}>Chat with CropAI about this project</Link>
          <Link className="button secondary" to={`/agents?project=${project.project_id}`}>Open agents dashboard</Link>
        </div>
      </section>

      <section className="panel project-upload">
        <h2>Add leaves or plants</h2>
        <label className="agents-file">
          <input type="file" multiple accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
          <span>{files.length ? `${files.length} photo${files.length === 1 ? "" : "s"} selected` : "Choose one or more photos"}</span>
        </label>
        <label className="project-check">
          <input type="checkbox" checked={useFilenames} onChange={(event) => setUseFilenames(event.target.checked)} />
          Label each photo with its file name (you can edit labels below)
        </label>
        <div className="agents-run-actions">
          <button className="button primary" disabled={!files.length || busy} onClick={() => void upload(files)}>
            {busy ? "Analysing…" : `Analyse and add ${files.length || ""}`.trim()}
          </button>
          <SavedPhotoPicker disabled={busy} onPick={(file) => void upload([file], false)} />
        </div>
        {progress && <small className="muted">{progress}</small>}
        {error && <p className="chat-error" role="alert">{error}</p>}
      </section>

      <section className="project-summary" aria-label="Diagnosis summary">
        <strong>{project.image_count} photo{project.image_count === 1 ? "" : "s"}</strong>
        {project.summary.map((item) => <span key={item.label} className="tag">{item.count} × {item.label}</span>)}
      </section>

      {project.images.length === 0 && <p className="muted">No photos in this project yet.</p>}
      <section className="history-grid">
        {project.images.map((image) => (
          <PhotoCard
            key={image.diagnosis_id}
            image={image}
            actions={<button type="button" className="link-button danger" onClick={() => void removeFromProject(image.diagnosis_id)}>Remove from project</button>}
          />
        ))}
      </section>

      <div className="history-toolbar project-danger">
        <button type="button" className="link-button danger" onClick={() => void removeProject()}>Delete project</button>
      </div>
    </main>
  );
}
