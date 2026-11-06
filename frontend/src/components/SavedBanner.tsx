import type { DiagnosisResponse } from "../types/api";

/** Shown when the backend returned a stored diagnosis because this exact photo was analysed before. */
export function SavedBanner({ result, onReanalyse, busy = false }: { result: DiagnosisResponse; onReanalyse?: () => void; busy?: boolean }) {
  if (!result.from_saved) return null;
  const when = result.saved_at ? new Date(result.saved_at).toLocaleString() : "earlier";
  return (
    <div className="saved-banner" role="status">
      <span>
        <strong>Saved diagnosis</strong> — this photo was analysed on {when}, so the stored result is shown
        instead of running the model again.
      </span>
      {onReanalyse && (
        <button type="button" className="button secondary" disabled={busy} onClick={onReanalyse}>
          {busy ? "Re-analysing…" : "Re-analyse"}
        </button>
      )}
    </div>
  );
}
