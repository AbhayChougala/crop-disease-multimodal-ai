import { type FormEvent, useState } from "react";

import { sendChatMessage } from "../api/client";

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: Array<{ source_id: string; title: string }>;
}

const suggestions = [
  "How reliable is this prediction?",
  "What does the severity estimate mean?",
  "What should I do if the result is uncertain?",
];

export function ChatPanel({ sessionId, isMock }: { sessionId: string; isMock: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const content = (suggested ?? message).trim();
    if (!content || loading) return;
    setMessages((items) => [...items, { role: "user", text: content }]);
    setMessage("");
    setLoading(true);
    setError(null);
    try {
      const response = await sendChatMessage(sessionId, content);
      setMessages((items) => [...items, { role: "assistant", text: response.answer, sources: response.sources }]);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Chat request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <article className="panel chat-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Diagnosis-aware follow-up</p>
          <h2>Ask about this result</h2>
        </div>
        <span className="tag">{isMock ? "Demo chat" : "Grounded chat"}</span>
      </div>
      <div className="suggestion-row">
        {suggestions.map((suggestion) => (
          <button key={suggestion} type="button" onClick={() => submit(undefined, suggestion)}>
            {suggestion}
          </button>
        ))}
      </div>
      <div className="conversation" aria-live="polite">
        {messages.length === 0 && (
          <p className="muted">Questions stay linked to diagnosis session {sessionId.slice(0, 8)}.</p>
        )}
        {messages.map((item, index) => (
          <div className={`message ${item.role}`} key={`${item.role}-${index}`}>
            <strong>{item.role === "user" ? "You" : "Assistant"}</strong>
            <p>{item.text}</p>
            {item.sources && item.sources.length > 0 && <small>{item.sources.map((source) => `${source.source_id}: ${source.title}`).join(" · ")}</small>}
          </div>
        ))}
        {loading && <p className="muted">Preparing a cautious response…</p>}
      </div>
      <form className="chat-form" onSubmit={(event) => submit(event)}>
        <label className="sr-only" htmlFor="follow-up">Follow-up question</label>
        <input id="follow-up" value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask a follow-up question" maxLength={2000} />
        <button className="button primary" disabled={!message.trim() || loading}>Send</button>
      </form>
      {error && <p className="chat-error" role="alert">{error}</p>}
      {isMock && <p className="muted">Demo responses contain no agricultural advice or citations.</p>}
    </article>
  );
}
