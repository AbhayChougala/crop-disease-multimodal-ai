import type { AgentStep, ChatResponse, FollowUp } from "../types/api";

interface AgentData {
  agent_trace?: AgentStep[];
  questions?: string[];
  follow_up?: FollowUp | null;
  risk_level?: string | null;
}

/** Agent fields travel in message metadata so they survive a page reload. */
export function agentMetadata(response: ChatResponse): Record<string, unknown> {
  return {
    ...response.metadata,
    agent_trace: response.agent_trace ?? [],
    questions: response.questions ?? [],
    follow_up: response.follow_up ?? null,
    risk_level: response.risk_level ?? null,
  };
}

const STATUS_LABEL: Record<AgentStep["status"], string> = {
  ok: "done",
  fallback: "rule-based",
  skipped: "skipped",
  error: "failed",
};

export function AgentInsights({
  metadata,
  onPickQuestion,
}: {
  metadata: Record<string, unknown>;
  onPickQuestion?: (question: string) => void;
}) {
  const data = metadata as AgentData;
  const trace = data.agent_trace ?? [];
  if (trace.length === 0) return null;
  const questions = data.questions ?? [];
  const followUp = data.follow_up;
  const agents = new Set(trace.filter((step) => step.action !== "finish").map((step) => step.agent));
  const usedLlm = trace.some((step) => step.decided_by === "llm");

  return (
    <div className="agent-insights">
      {questions.length > 0 && (
        <div className="agent-questions">
          <small>Tap a question to answer it:</small>
          {questions.map((question) => (
            <button key={question} type="button" onClick={() => onPickQuestion?.(question)}>
              {question}
            </button>
          ))}
        </div>
      )}
      {(data.risk_level || followUp) && (
        <div className="agent-cards">
          {data.risk_level && (
            <span className={`tag risk-${data.risk_level}`}>Spread risk: {data.risk_level}</span>
          )}
          {followUp && (
            <span className="tag">
              Re-check on {new Date(followUp.due_date).toLocaleDateString([], { day: "numeric", month: "short" })}
            </span>
          )}
        </div>
      )}
      <details className="agent-trace">
        <summary>
          How CropAI worked this out · {agents.size} agent{agents.size === 1 ? "" : "s"}
          {usedLlm ? " · AI orchestrated" : " · rule-based"}
        </summary>
        <ol>
          {trace.map((step, index) => (
            <li key={`${step.action}-${index}`} className={`agent-step status-${step.status}`}>
              <strong>{step.agent}</strong>
              <span className="agent-step-status">{STATUS_LABEL[step.status]}</span>
              <p>{step.summary}</p>
              <small>
                {step.decided_by === "llm" ? "Chosen by orchestrator" : "Chosen by rule"}
                {step.reason ? ` — ${step.reason}` : ""}
                {step.ms ? ` · ${Math.round(step.ms)} ms` : ""}
              </small>
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}
