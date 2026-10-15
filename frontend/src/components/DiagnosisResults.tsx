import type { DiagnosisResponse } from "../types/api";
import { ChatPanel } from "./ChatPanel";
import { DemoBanner } from "./DemoBanner";

export function DiagnosisResults({ result }: { result: DiagnosisResponse }) {
  const artifacts = [
    ["Original", result.artifacts.original_image],
    ["Preprocessed", result.artifacts.preprocessed_image],
    ["Leaf mask", result.artifacts.leaf_mask],
    ["Leaf isolated", result.artifacts.leaf_isolated],
    ["Experimental lesion mask", result.artifacts.lesion_mask],
    ["Experimental lesion overlay", result.artifacts.lesion_overlay],
    ["Grad-CAM", result.artifacts.gradcam],
  ].filter((entry): entry is [string, string] => Boolean(entry[1]));

  return (
    <section className="results-stack" aria-live="polite">
      {result.is_mock && <DemoBanner />}
      <div className="metric-grid">
        <article className="metric-card">
          <span>{result.is_mock ? "Mock prediction" : "Model prediction"}</span>
          <strong>{result.prediction.disease ?? "Abstained"}</strong>
          <small>{result.prediction.crop ?? result.prediction.abstention_reason}</small>
        </article>
        <article className="metric-card">
          <span>{result.is_mock ? "Placeholder confidence" : "Confidence"}</span>
          <strong>{(result.prediction.confidence * 100).toFixed(1)}%</strong>
          <small>{result.prediction.architecture} · {result.evidence.model_mode}</small>
        </article>
        <article className="metric-card warning-card">
          <span>Experimental severity</span>
          <strong>{result.severity.category}</strong>
          <small>
            {result.severity.affected_area_percent === null
              ? "Unavailable"
              : `${result.severity.affected_area_percent.toFixed(1)}% color-mask area`}
          </small>
        </article>
      </div>

      <article className="panel">
        <div className="section-heading"><div><p className="eyebrow">Classifier ranking</p><h2>Top predictions</h2></div><span className="tag">{result.evidence.model_id}</span></div>
        <ol>{result.prediction.top_k.map((item) => <li key={item.label}>{item.label} — {(item.confidence * 100).toFixed(1)}%</li>)}</ol>
      </article>

      <article className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Visual evidence</p>
            <h2>Pipeline artifacts</h2>
          </div>
          <span className="tag">Not ground truth</span>
        </div>
        <div className="artifact-grid">
          {artifacts.map(([label, source]) => (
            <figure key={label}>
              <img src={source} alt={label} />
              <figcaption>{label}</figcaption>
            </figure>
          ))}
        </div>
      </article>

      <article className="panel explanation">
        <p className="eyebrow">{result.advisory.is_mock ? "Mock explanation" : "Grounded explanation"}</p>
        <h2>Decision-support summary</h2>
        <p>{result.advisory.summary}</p>
        <div className="callout">{result.advisory.when_to_seek_expert_help}</div>
        {result.advisory.sources.length === 0 && (
          <p className="muted">No retrieved sources support this explanation.</p>
        )}
        {result.advisory.sources.length > 0 && <ul>{result.advisory.sources.map((source) => <li key={source.source_id}>{source.source_id}: {source.title}</li>)}</ul>}
      </article>

      <article className="panel">
        <p className="eyebrow">Measured runtime</p>
        <h2>Stage timings</h2>
        <div className="timing-row">
          {Object.entries(result.timings_ms).map(([name, value]) => (
            <div key={name}>
              <span>{name}</span>
              <strong>{value.toFixed(2)} ms</strong>
            </div>
          ))}
        </div>
      </article>

      <ChatPanel sessionId={result.session_id} isMock={result.is_mock} />

      {result.warnings.length > 0 && (
        <article className="panel warnings">
          <h2>Warnings and limitations</h2>
          <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </article>
      )}
    </section>
  );
}
