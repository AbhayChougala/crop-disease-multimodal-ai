import { useState } from "react";

import { listStoredImages, loadStoredImageFile, storedImageUrl } from "../api/client";
import type { StoredImage } from "../types/api";

/** "Choose from saved photos": pick a leaf photo already stored in the local database. */
export function SavedPhotoPicker({ onPick, disabled = false }: { onPick: (file: File) => void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [images, setImages] = useState<StoredImage[] | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function toggle() {
    if (open) { setOpen(false); return; }
    setOpen(true); setError(null);
    try {
      setImages((await listStoredImages()).images);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not load saved photos.");
    }
  }

  async function pick(image: StoredImage) {
    setLoadingId(image.diagnosis_id); setError(null);
    try {
      onPick(await loadStoredImageFile(image));
      setOpen(false);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not load that photo.");
    } finally { setLoadingId(null); }
  }

  return (
    <div className="saved-picker">
      <button type="button" className="button secondary" disabled={disabled} onClick={() => void toggle()} aria-expanded={open}>
        {open ? "Close saved photos" : "Choose from saved photos"}
      </button>
      {open && (
        <div className="saved-picker-panel">
          {error && <p className="chat-error" role="alert">{error}</p>}
          {!images && !error && <p className="muted">Loading…</p>}
          {images?.length === 0 && <p className="muted">No saved photos yet. Every photo you analyse is saved here.</p>}
          <div className="saved-picker-grid">
            {images?.map((image) => (
              <button
                type="button"
                key={image.diagnosis_id}
                className="saved-picker-item"
                disabled={loadingId !== null}
                onClick={() => void pick(image)}
                title={image.filename}
              >
                <img src={storedImageUrl(image.diagnosis_id)} alt={image.filename} loading="lazy" />
                <span>{loadingId === image.diagnosis_id ? "Loading…" : image.disease ? `${image.crop} – ${image.disease}` : image.filename}</span>
                <small>{new Date(image.created_at).toLocaleDateString()}</small>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
