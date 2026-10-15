import { Link } from "react-router-dom";

export function HomePage() {
  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">B.Tech AI/ML research prototype</p>
          <h1>Crop Disease Detection and Diagnosis Using Multimodal AI</h1>
          <p className="hero-lead">
            A reproducible pipeline connecting OpenCV evidence, deep-learning classification,
            explainability, and knowledge-grounded guidance.
          </p>
          <div className="button-row">
            <Link className="button primary" to="/diagnose">Analyze an image</Link>
            <Link className="button secondary" to="/research">View research design</Link>
          </div>
        </div>
        <div className="pipeline-visual" aria-label="System pipeline">
          {[
            ["01", "Observe", "Upload or camera frame"],
            ["02", "Analyze", "CV, classifier, explainability"],
            ["03", "Ground", "Retrieval and cited context"],
            ["04", "Explain", "Decision-support response"],
          ].map(([number, title, text]) => (
            <div className="pipeline-step" key={number}>
              <span>{number}</span><div><strong>{title}</strong><small>{text}</small></div>
            </div>
          ))}
        </div>
      </section>
      <section className="feature-grid">
        <article><span className="feature-icon">CV</span><h2>Computer Vision</h2><p>Configurable preprocessing and independent heuristic masks support controlled ablation studies.</p></article>
        <article><span className="feature-icon">XAI</span><h2>Explainable AI</h2><p>Grad-CAM will show classifier influence without being misrepresented as lesion ground truth.</p></article>
        <article><span className="feature-icon">RAG</span><h2>Grounded Guidance</h2><p>Retrieved agricultural sources will remain separate from evidence actually observed by the model.</p></article>
      </section>
      <section className="disclaimer">
        <strong>Research status</strong>
        <p>The current build is a mock-mode software prototype. It is not a diagnostic product and must not guide treatment decisions.</p>
      </section>
    </main>
  );
}
