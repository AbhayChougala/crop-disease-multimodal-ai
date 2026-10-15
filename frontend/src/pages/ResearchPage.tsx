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
  }, []);

  return <main className="page-shell"><section className="page-intro"><p className="eyebrow">Research methodology</p><h1>Experiments before claims</h1><p>{message}</p></section><div className="study-grid">{studies.map((study, index) => <article key={study.id}><span>{String(index + 1).padStart(2, "0")}</span><h2>{study.title}</h2><p>{study.protocol}</p><small>{study.status === "complete" ? `${study.measurements.length} measured artifact(s)` : study.status === "failed" ? "Experiment artifacts need attention" : "Pending experiment"}</small></article>)}</div></main>;
}
