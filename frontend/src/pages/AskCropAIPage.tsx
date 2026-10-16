import { type FormEvent, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { createChatSession, deleteChatSession, sendSessionMessage } from "../api/client";
import type { ChatCitation, ChatMessage, ChatResponse, ChatSession } from "../types/api";

const starterQuestions = [
  "What evidence should I collect before asking an expert?",
  "How should I interpret an uncertain prediction?",
  "What image quality issues can affect a crop diagnosis?",
];

function responseMessage(response: ChatResponse): ChatMessage {
  return {
    message_id: response.message_id,
    role: "assistant",
    content: response.answer,
    citations: response.sources,
    metadata: response.metadata,
    created_at: response.created_at,
  };
}

export function AskCropAIPage() {
  const [parameters] = useSearchParams();
  const diagnosisId = parameters.get("diagnosis") ?? undefined;
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [citations, setCitations] = useState<ChatCitation[]>([]);
  const [language, setLanguage] = useState<"en" | "kn" | "hi">("en");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<ChatResponse["metadata"] | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    createChatSession(diagnosisId ? "diagnosis" : "general", diagnosisId, language)
      .then((created) => {
        if (!active) return;
        setSession(created);
        setMessages(created.messages);
        const latest = [...created.messages].reverse().find((item) => item.citations.length > 0);
        setCitations(latest?.citations ?? []);
      })
      .catch((problem) => {
        if (active) setError(problem instanceof Error ? problem.message : "Could not start chat.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [diagnosisId]);

  const modeLabel = session?.mode === "diagnosis" ? "Diagnosis-aware" : "General crop Q&A";
  const latestAssistant = useMemo(
    () => [...messages].reverse().find((item) => item.role === "assistant"),
    [messages],
  );

  async function submit(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const question = (suggested ?? draft).trim();
    if (!session || !question || sending) return;
    const now = new Date().toISOString();
    setMessages((items) => [...items, {
      message_id: `pending-${now}`,
      role: "user",
      content: question,
      citations: [],
      metadata: {},
      created_at: now,
    }]);
    setDraft("");
    setLastQuestion(question);
    setSending(true);
    setError(null);
    try {
      const response = await sendSessionMessage(session.session_id, question, language);
      setMessages((items) => [...items, responseMessage(response)]);
      setCitations(response.sources);
      setMetadata(response.metadata);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Message failed.");
    } finally {
      setSending(false);
    }
  }

  async function resetChat() {
    if (session) await deleteChatSession(session.session_id).catch(() => undefined);
    setLoading(true);
    setMessages([]);
    setCitations([]);
    setMetadata(null);
    setError(null);
    try {
      const created = await createChatSession("general", undefined, language);
      setSession(created);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not reset chat.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell ask-page">
      <section className="page-intro ask-intro">
        <div><p className="eyebrow">Grounded agricultural assistant</p><h1>Ask CropAI</h1><p>Answers use retrieved sources when available and abstain when evidence is missing.</p></div>
        <div className="chat-toolbar">
          <label>Language scaffold<select value={language} onChange={(event) => setLanguage(event.target.value as "en" | "kn" | "hi")}><option value="en">English</option><option value="kn">ಕನ್ನಡ</option><option value="hi">हिन्दी</option></select></label>
          <button className="button secondary" type="button" onClick={resetChat}>New general chat</button>
        </div>
      </section>

      <div className="ask-layout">
        <section className="panel ask-conversation">
          <div className="chat-status"><span className="status-dot" /> <strong>{modeLabel}</strong><small>{session ? `Session ${session.session_id.slice(0, 8)}` : "Starting session…"}</small></div>
          {session?.mode === "diagnosis" && <div className="callout">Visual evidence from diagnosis {session.diagnosis_id?.slice(0, 8)} is attached. Retrieved knowledge remains separate from model evidence.</div>}
          <div className="suggestion-row">{starterQuestions.map((question) => <button key={question} type="button" disabled={!session || sending} onClick={() => submit(undefined, question)}>{question}</button>)}</div>
          <div className="conversation ask-thread" aria-live="polite">
            {!loading && messages.length === 0 && <div className="empty-chat"><strong>Ask a focused crop-health question.</strong><p>Include the crop, visible symptoms, location context, and uncertainty. Do not include secrets or personal information.</p></div>}
            {messages.map((item) => <article className={`message ${item.role}`} key={item.message_id}><div className="message-heading"><strong>{item.role === "user" ? "You" : "CropAI"}</strong><time>{new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div><p>{item.content}</p>{item.citations.length > 0 && <button type="button" className="citation-link" onClick={() => setCitations(item.citations)}>{item.citations.length} cited passage{item.citations.length === 1 ? "" : "s"}</button>}{item.role === "assistant" && <button type="button" className="copy-link" onClick={() => navigator.clipboard.writeText(item.content)}>Copy</button>}</article>)}
            {sending && <div className="message assistant loading-message">Retrieving evidence and preparing a cautious response…</div>}
          </div>
          <form className="ask-form" onSubmit={submit}><label className="sr-only" htmlFor="ask-message">Question</label><textarea id="ask-message" value={draft} onChange={(event) => setDraft(event.target.value)} maxLength={2000} placeholder="Ask about crop symptoms, uncertainty, prevention evidence…" /><button className="button primary" disabled={!draft.trim() || sending || !session}>Send</button></form>
          {error && <div className="chat-error" role="alert">{error}{lastQuestion && <button type="button" onClick={() => submit(undefined, lastQuestion)}>Retry</button>}</div>}
          <p className="muted chat-disclaimer">Decision support only. Verify important actions with a qualified agricultural expert.</p>
        </section>

        <aside className="source-drawer">
          <div className="panel"><p className="eyebrow">Grounding</p><h2>Sources</h2>{citations.length === 0 ? <p className="muted">No source is attached to the selected answer. Treat it as unsupported or insufficient evidence.</p> : citations.map((source) => <article className="source-card" key={source.source_id}><span>{source.source_id}</span><h3>{source.title}</h3><p>{source.excerpt}</p><small>{[source.organization, source.section_name, source.page ? `page ${source.page}` : null].filter(Boolean).join(" · ")}</small>{source.url && <a href={source.url} target="_blank" rel="noreferrer">Open source ↗</a>}</article>)}</div>
          <div className="panel pipeline-meta"><p className="eyebrow">Pipeline metadata</p><dl><div><dt>Mode</dt><dd>{modeLabel}</dd></div><div><dt>Provider</dt><dd>{metadata?.provider ?? session?.provider ?? "—"}</dd></div><div><dt>Model</dt><dd>{metadata?.model || session?.model || "—"}</dd></div><div><dt>Retriever</dt><dd>{metadata?.retriever ?? "—"}</dd></div><div><dt>Retrieved</dt><dd>{metadata?.retrieved_chunks ?? 0} chunks</dd></div><div><dt>Total time</dt><dd>{metadata ? `${metadata.total_ms.toFixed(0)} ms` : "—"}</dd></div></dl>{latestAssistant?.citations.length === 0 && <span className="tag caution-tag">No citation</span>}</div>
        </aside>
      </div>
    </main>
  );
}
