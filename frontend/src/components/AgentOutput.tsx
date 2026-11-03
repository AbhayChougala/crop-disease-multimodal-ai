import type { AgentName } from "../types/api";

/** Readable view of one agent's structured result (shapes mirror backend crop_ai/agents/state.py). */

type Data = Record<string, unknown>;

const list = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
const text = (value: unknown) => (typeof value === "string" ? value : "");

function Bullets({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="agent-output-block">
      <h4>{title}</h4>
      <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul>
    </div>
  );
}

function Evidence({ data }: { data: Data }) {
  const passages = (data.passages as Data[] | undefined) ?? [];
  if (passages.length === 0) return <p className="muted">No fact sheets found. Build the knowledge base (see README).</p>;
  return (
    <ol className="agent-passages">
      {passages.map((item) => (
        <li key={text(item.citation_id)}>
          <strong>{text(item.title)}</strong>
          <small>{text(item.organization)} · {text(item.citation_id)} · relevance {String(item.score)}</small>
          <p>{text(item.excerpt)}…</p>
        </li>
      ))}
    </ol>
  );
}

function Diagnosis({ data }: { data: Data }) {
  const candidates = (data.candidates as Data[] | undefined) ?? [];
  return (
    <>
      <p className="agent-output-lead">{text(data.summary)}</p>
      <div className="agent-cards">
        <span className="tag">Image model: {text(data.agreement_with_model).replace("_", " ")}</span>
        <span className={`tag ${data.is_ambiguous ? "risk-moderate" : "risk-low"}`}>
          {data.is_ambiguous ? "Ambiguous — ask questions" : "Clear"}
        </span>
      </div>
      <div className="agent-candidates">
        {candidates.map((item) => (
          <article key={`${text(item.crop)}-${text(item.disease)}`}>
            <strong>{text(item.crop)} – {text(item.disease)}</strong>
            <span className={`tag likelihood-${text(item.likelihood)}`}>{text(item.likelihood)}</span>
            <Bullets title="Supporting signs" items={list(item.supporting_signs)} />
            <Bullets title="Conflicting signs" items={list(item.conflicting_signs)} />
          </article>
        ))}
      </div>
      <Bullets title="Signs that would tell them apart" items={list(data.distinguishing_signs)} />
    </>
  );
}

function Risk({ data }: { data: Data }) {
  const weather = data.weather as Record<string, number> | null;
  return (
    <>
      <div className="agent-cards">
        <span className={`tag risk-${text(data.level)}`}>Spread risk: {text(data.level)}</span>
        <span className="tag">Score {String(data.score)}/100</span>
      </div>
      <Bullets title="Why" items={list(data.factors)} />
      {weather && (
        <div className="agent-output-block">
          <h4>3-day forecast (Open-Meteo)</h4>
          <dl className="agent-weather">
            {Object.entries(weather).map(([key, value]) => (
              <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{Math.round(value * 10) / 10}</dd></div>
            ))}
          </dl>
        </div>
      )}
      <small className="muted">{text(data.method)} — not validated against field outcomes.</small>
    </>
  );
}

const CATEGORIES = [
  ["immediate", "Do now"],
  ["cultural", "Field practices"],
  ["organic", "Organic options"],
  ["chemical", "Chemical options (follow the label and local rules)"],
] as const;

function Treatment({ data }: { data: Data }) {
  const items = (data.items as Data[] | undefined) ?? [];
  return (
    <>
      <p className="agent-output-lead">{text(data.summary)}</p>
      {items.length === 0 && <p className="muted">No grounded actions were produced.</p>}
      {CATEGORIES.map(([key, title]) => (
        <Bullets
          key={key}
          title={title}
          items={items
            .filter((item) => item.category === key)
            .map((item) => {
              const ids = list(item.source_ids);
              return ids.length ? `${text(item.action)} [${ids.join(", ")}]` : text(item.action);
            })}
        />
      ))}
      <Bullets title="Safety" items={list(data.safety_notes)} />
    </>
  );
}

function FollowUp({ data }: { data: Data }) {
  const due = text(data.due_date);
  return (
    <>
      <p className="agent-output-lead">
        Re-check in {String(data.reinspect_in_days)} days
        {due && ` (${new Date(due).toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })})`}
      </p>
      <Bullets title="Look for" items={list(data.what_to_check)} />
      <Bullets title="Call an expert if" items={list(data.escalate_if)} />
      {text(data.reassessment) && <p className="callout">{text(data.reassessment)}</p>}
    </>
  );
}

export function AgentOutput({ agent, data }: { agent: AgentName; data: Data | null }) {
  if (!data) return <p className="muted">This agent produced no result. See the steps below.</p>;
  switch (agent) {
    case "retrieve_evidence": return <Evidence data={data} />;
    case "diagnose": return <Diagnosis data={data} />;
    case "ask_farmer":
      return (
        <>
          <ol className="agent-output-questions">{list(data.questions).map((q) => <li key={q}>{q}</li>)}</ol>
          <p className="muted">Type your answers in the box, then run the Doctor (or any agent) again.</p>
        </>
      );
    case "assess_risk": return <Risk data={data} />;
    case "recommend_treatment": return <Treatment data={data} />;
    case "plan_followup": return <FollowUp data={data} />;
    case "answer_question": return <p className="agent-output-answer">{text(data.answer)}</p>;
  }
}
