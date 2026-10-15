export function ResearchPage() {
  const studies = [
    ["Model study", "ResNet50 vs EfficientNet-B0 vs MobileNetV2"],
    ["CV ablation", "Raw vs standard preprocessing vs segmented input"],
    ["Domain study", "Controlled dataset vs field imagery"],
    ["Grounding study", "LLM without retrieval vs RAG + LLM"],
    ["Runtime study", "Static diagnosis vs live-frame latency"],
    ["XAI study", "Representative and failure-case Grad-CAM review"],
  ];
  return <main className="page-shell"><section className="page-intro"><p className="eyebrow">Research methodology</p><h1>Experiments before claims</h1><p>No experimental metrics exist yet. This page intentionally shows the protocol, not invented charts.</p></section><div className="study-grid">{studies.map(([title, text], index) => <article key={title}><span>{String(index + 1).padStart(2, "0")}</span><h2>{title}</h2><p>{text}</p><small>Pending real data and validated model artifacts</small></article>)}</div></main>;
}
