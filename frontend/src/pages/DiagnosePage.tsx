import { useEffect, useState } from "react";

import { diagnoseImage } from "../api/client";
import { DiagnosisResults } from "../components/DiagnosisResults";
import type { DiagnosisResponse } from "../types/api";

export function DiagnosePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function analyze() {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult(await diagnoseImage(file)); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Analysis failed."); }
    finally { setLoading(false); }
  }

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow">Static image workflow</p>
        <h1>Inspect a leaf image</h1>
        <p>Upload a clear JPEG, PNG, or WebP image. This build runs a real OpenCV pipeline and an explicitly mocked classifier.</p>
      </section>
      <section className="upload-layout">
        <label className={`dropzone ${preview ? "has-image" : ""}`}>
          {preview ? <img src={preview} alt="Selected leaf preview" /> : <><span className="upload-glyph">+</span><strong>Choose a leaf image</strong><small>JPEG, PNG, or WebP · maximum 10 MB</small></>}
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
        </label>
        <aside className="upload-actions">
          <span className="status-pill">Demo model active</span>
          <h2>Analysis pipeline</h2>
          <ol><li>Validate and decode</li><li>Preprocess and isolate leaf</li><li>Mock classification</li><li>Experimental severity</li><li>Mock advisory</li></ol>
          <button className="button primary" disabled={!file || loading} onClick={analyze}>{loading ? "Analyzing…" : "Run analysis"}</button>
          {error && <p className="error" role="alert">{error}</p>}
        </aside>
      </section>
      {result && <DiagnosisResults result={result} />}
    </main>
  );
}
