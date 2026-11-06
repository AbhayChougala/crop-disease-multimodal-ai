import { Link } from "react-router-dom";

import { storedImageUrl } from "../api/client";
import type { AgentMemory, ProjectDetail, StoredImage } from "../types/api";

const CHECKS = [
  ["has_diagnosis", "Doctor"],
  ["has_risk", "Risk"],
  ["has_treatment", "Treatment"],
  ["has_follow_up", "Follow-up"],
] as const;

export const photoName = (image: StoredImage) => image.note || image.filename;

/** Every photo of a project with what the agents already did for it; click one to work on it. */
export function ProjectAgentDashboard({
  project,
  memories,
  selectedId,
  onSelect,
}: {
  project: ProjectDetail;
  memories: Record<string, AgentMemory | null>;
  selectedId: string | null;
  onSelect: (image: StoredImage) => void;
}) {
  return (
    <section className="panel project-dashboard" aria-label={`Agents dashboard for ${project.name}`}>
      <div className="section-heading">
        <h2>{project.name}: agents dashboard</h2>
        <span className="muted">
          {project.image_count} photo{project.image_count === 1 ? "" : "s"} · <Link to={`/projects/${project.project_id}`}>add photos</Link> · <Link to={`/ask?project=${project.project_id}`}>chat about this project</Link>
        </span>
      </div>
      <div className="project-summary">{project.summary.map((item) => <span key={item.label} className="tag">{item.count} × {item.label}</span>)}</div>
      {project.images.length === 0 && <p className="muted">This project has no photos yet. Add some on the project page.</p>}
      <div className="dashboard-grid" role="listbox" aria-label="Choose a photo to work on">
        {project.images.map((image) => {
          const memory = memories[image.diagnosis_id];
          return (
            <button
              type="button"
              role="option"
              aria-selected={selectedId === image.diagnosis_id}
              key={image.diagnosis_id}
              className={`dashboard-photo ${selectedId === image.diagnosis_id ? "selected" : ""}`}
              onClick={() => onSelect(image)}
            >
              <img src={storedImageUrl(image.diagnosis_id)} alt={photoName(image)} loading="lazy" />
              <span className="dashboard-photo-text">
                <strong>{photoName(image)}</strong>
                <small>{image.crop && image.disease ? `${image.crop} – ${image.disease}` : "No confident prediction"}{image.confidence !== null ? ` · ${Math.round(image.confidence * 100)}%` : ""}</small>
                <span className="dashboard-checks">
                  {memory === undefined
                    ? <small className="muted">…</small>
                    : memory === null
                      ? <small className="muted">diagnosis unavailable</small>
                      : CHECKS.map(([key, label]) => <span key={key} className={`tag ${memory[key] ? "risk-low" : ""}`}>{memory[key] ? "✓" : "–"} {label}</span>)}
                  {memory && memory.pending_questions.length > 0 && <span className="tag risk-moderate">? questions</span>}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
