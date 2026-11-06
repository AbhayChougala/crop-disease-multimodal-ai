import { useState } from "react";
import { Link } from "react-router-dom";

import { storedImageUrl, updateStoredImage } from "../api/client";
import type { StoredImage } from "../types/api";

const size = (bytes: number) => (bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`);

/** One stored photo with its diagnosis and an editable label; click the photo for the full diagnosis. */
export function PhotoCard({
  image,
  projectName,
  onChanged,
  actions,
}: {
  image: StoredImage;
  projectName?: string;
  onChanged?: (image: StoredImage) => void;
  actions?: React.ReactNode;
}) {
  const [note, setNote] = useState(image.note);
  const [state, setState] = useState<"idle" | "saving" | "saved" | "error">("idle");

  async function saveNote() {
    if (note.trim() === image.note) return;
    setState("saving");
    try {
      const updated = await updateStoredImage(image.diagnosis_id, { note });
      setState("saved");
      onChanged?.(updated);
    } catch {
      setState("error");
    }
  }

  return (
    <article className="history-card">
      <Link to={`/history/${image.diagnosis_id}`} title="Open the saved diagnosis">
        <img src={storedImageUrl(image.diagnosis_id)} alt={image.note || image.filename} loading="lazy" />
      </Link>
      <div>
        <strong>{image.crop && image.disease ? `${image.crop} – ${image.disease}` : "No confident prediction"}</strong>
        {image.confidence !== null && <span>{Math.round(image.confidence * 100)}% image-model confidence</span>}
        {image.is_mock && <span className="tag caution-tag">DEMO MODE — placeholder prediction</span>}
        {projectName && <span className="tag">Project: {projectName}</span>}
        <input
          className="photo-label"
          value={note}
          maxLength={300}
          placeholder="Add a label, e.g. Plant 3, row 2"
          aria-label="Photo label"
          onChange={(event) => { setNote(event.target.value); setState("idle"); }}
          onBlur={() => void saveNote()}
          onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }}
        />
        {state !== "idle" && <small>{state === "saving" ? "Saving…" : state === "saved" ? "Label saved" : "Could not save the label"}</small>}
        <small>{new Date(image.created_at).toLocaleString()} · {image.filename} · {size(image.size_bytes)}</small>
        <div className="photo-card-actions">
          <Link to={`/history/${image.diagnosis_id}`}>View diagnosis</Link>
          {actions}
        </div>
      </div>
    </article>
  );
}
