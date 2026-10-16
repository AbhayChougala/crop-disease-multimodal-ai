import { useEffect, useState } from "react";

type Study = {
  id: string;
  title: string;
  protocol: string;
  status: "pending" | "complete" | "failed";
  measurements: Array<{ artifact: string; metrics: Record<string, number> }>;
};

const fallback: Study[] = [
  ["model_comparison", "Model comparison", "ResNet50 vs EfficientNet-B0 vs MobileNetV2"],
  ["preprocessing_ablation", "CV ablation", "Raw vs standard vs leaf-isolated input"],
  ["cross_domain", "Domain study", "Controlled images vs mapped field imagery"],
  ["llm_grounding", "Grounding study", "No retrieval vs RAG on fixed questions"],
  ["runtime", "Runtime study", "Measured stage-level latency by device"],
  ["explainability", "XAI study", "Representative and failure-case Grad-CAM review"],
].map(([id, title, protocol]) => ({
  id,
  title,
  protocol,
  status: "pending",
  measurements: [],
}));

export function ResearchPage() {
  const [studies, setStudies] = useState<Study[]>(fallback);
  const [message, setMessage] = useState("Loading experiment registry…");
  const [rag, setRag] = useState<{ available: boolean; retriever: string; indexed_documents: number; indexed_chunks: number; embedding_model: string | null; error: string | null } | null>(null);
  const [llm, setLlm] = useState<{ status: { provider: string; model: string; state: string; detail: string | null }; capabilities: { streaming: boolean; structured_output: boolean } } | null>(null);
  const [ragResults, setRagResults] = useState<Array<{ file: string; evaluation_status: string; retriever: string; case_count: number; top_k: number; precision_at_k: number; recall_at_k: number; hit_rate_at_k: number; mean_reciprocal_rank: number; source_coverage: number }>>([]);

  useEffect(() => {
    const base = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
    fetch(`${base}/research/summary`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("API unavailable")))
      .then((payload) => {
        if (Array.isArray(payload.studies)) setStudies(payload.studies);
        setMessage(payload.status === "results_available"
          ? "Measured results imported."
          : "No validated experiment metrics exist yet.");
      })
      .catch(() => setMessage("Experiment API unavailable; showing the planned protocol."));
    fetch(`${base}/rag/status`).then((response) => response.ok ? response.json() : null).then(setRag).catch(() => undefined);
    fetch(`${base}/llm/status`).then((response) => response.ok ? response.json() : null).then(setLlm).catch(() => undefined);
    fetch(`${base}/research/rag-results`).then((response) => response.ok ? response.json() : { results: [] }).then((payload) => setRagResults(payload.results ?? [])).catch(() => undefined);
  }, []);

  return <main className="page-shell"><section className="page-intro"><p className="eyebrow">Research methodology</p><h1>Experiments before claims</h1><p>{message}</p></section><div className="runtime-grid"><article className="panel"><p className="eyebrow">Retrieval runtime</p><h2>{rag?.available ? `${rag.retriever} ready` : "RAG unavailable"}</h2><p>{rag ? `${rag.indexed_documents} source documents · ${rag.indexed_chunks} chunks` : "Status unavailable"}</p><small>{rag?.embedding_model ?? rag?.error ?? "TF-IDF baseline or not configured"}</small></article><article className="panel"><p className="eyebrow">Language runtime</p><h2>{llm ? `${llm.status.provider} · ${llm.status.state}` : "LLM status unavailable"}</h2><p>{llm?.status.model || "No configured model"}</p><small>{llm ? `${llm.capabilities.structured_output ? "Structured output" : "Unstructured output"} · ${llm.capabilities.streaming ? "streaming" : "non-streaming"}` : "—"}</small></article></div>{ragResults.length > 0 && <section className="panel"><p className="eyebrow">Saved retrieval evaluations</p><h2>Measured artifacts</h2><div className="result-table">{ragResults.map((result) => <article key={result.file}><strong>{result.retriever} · {result.evaluation_status}</strong><span>{result.case_count} cases · k={result.top_k}</span><span>Precision {result.precision_at_k.toFixed(3)} · Recall {result.recall_at_k.toFixed(3)} · Hit {result.hit_rate_at_k.toFixed(3)} · MRR {result.mean_reciprocal_rank.toFixed(3)}</span>{result.evaluation_status === "SAMPLE_FIXTURE_ONLY" && <small>Software fixture only — not research evidence.</small>}</article>)}</div></section>}<div className="study-grid">{studies.map((study, index) => <article key={study.id}><span>{String(index + 1).padStart(2, "0")}</span><h2>{study.title}</h2><p>{study.protocol}</p><small>{study.status === "complete" ? `${study.measurements.length} measured artifact(s)` : study.status === "failed" ? "Experiment artifacts need attention" : "Pending experiment"}</small></article>)}</div></main>;
}
