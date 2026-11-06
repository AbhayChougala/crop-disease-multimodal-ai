import { useEffect, useState } from "react";

import { deleteStoredImage, listProjects, listStoredImages } from "../api/client";
import { PhotoCard } from "../components/PhotoCard";
import type { Project, StoredImageList } from "../types/api";

/** Every leaf photo analysed on this computer, read from the backend's local SQLite database. */
export function HistoryPage() {
  const [data, setData] = useState<StoredImageList | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const [images, projectList] = await Promise.all([listStoredImages(), listProjects()]);
      setData(images);
      setProjects(projectList);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not load stored photos. Is the backend running?");
    }
  }

  useEffect(() => { void load(); }, []);

  async function remove(diagnosisId: string) {
    if (!window.confirm("Delete this photo from the local database?")) return;
    try {
      await deleteStoredImage(diagnosisId);
      await load();
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Delete failed.");
    }
  }

  const projectName = (id: string | null) => projects.find((project) => project.project_id === id)?.name;

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow">Local database</p>
        <h1>Photo history</h1>
        <p>Every photo analysed on this computer is saved with its diagnosis (backend/artifacts/runtime/chat.sqlite3). Click a photo to see its full saved diagnosis, give it a label, or group photos in a project.</p>
      </section>

      <div className="history-toolbar">
        <button type="button" className="button secondary" onClick={() => void load()}>Refresh</button>
        {data && <span className="muted">{data.total} photo{data.total === 1 ? "" : "s"} stored</span>}
        {data && !data.enabled && <span className="tag caution-tag">Saving is off (STORE_UPLOADED_IMAGES=false)</span>}
      </div>
      {error && <p className="chat-error" role="alert">{error}</p>}
      {data && data.images.length === 0 && <p className="muted">No photos yet. Analyse a leaf on the Diagnose page and it will appear here.</p>}

      <section className="history-grid">
        {data?.images.map((item) => (
          <PhotoCard
            key={item.diagnosis_id}
            image={item}
            projectName={projectName(item.project_id)}
            actions={<button type="button" className="link-button danger" onClick={() => void remove(item.diagnosis_id)}>Delete</button>}
          />
        ))}
      </section>
    </main>
  );
}
