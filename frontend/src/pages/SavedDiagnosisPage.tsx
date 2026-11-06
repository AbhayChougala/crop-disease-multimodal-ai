import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { loadSavedDiagnosis } from "../api/client";
import { DiagnosisResults } from "../components/DiagnosisResults";
import type { DiagnosisResponse } from "../types/api";

/** The full stored diagnosis of one saved photo (History → click a photo). Nothing is re-run. */
export function SavedDiagnosisPage() {
  const { diagnosisId = "" } = useParams();
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null); setError(null);
    loadSavedDiagnosis(diagnosisId)
      .then(setResult)
      .catch((problem) => setError(problem instanceof Error ? problem.message : "Could not load this diagnosis."));
  }, [diagnosisId]);

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow"><Link to="/history">← Photo history</Link></p>
        <h1>Saved diagnosis</h1>
        <p>This is the diagnosis stored when the photo was analysed. The model was not run again.</p>
      </section>
      {error && <p className="chat-error" role="alert">{error}</p>}
      {!result && !error && <p className="muted">Loading…</p>}
      {result && <DiagnosisResults result={result} />}
    </main>
  );
}
