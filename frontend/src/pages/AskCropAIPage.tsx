import { type FormEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { createChatSession, deleteChatSession, diagnoseImage, reassessSession, sendSessionMessage } from "../api/client";
import { AgentInsights, agentMetadata } from "../components/AgentInsights";
import type { ChatCitation, ChatMessage, ChatResponse, ChatSession, GeoLocation } from "../types/api";

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
    metadata: agentMetadata(response),
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
  const [warnings, setWarnings] = useState<string[]>([]);
  const [location, setLocation] = useState<GeoLocation | null>(null);
  const [locationNote, setLocationNote] = useState<string | null>(null);
  const photoRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  // Grow the input with its content (pasted prompts included); CSS max-height caps it.
  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight + 2}px`;
  }, [draft]);

  // Keep the newest message in view.
  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

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
      const response = await sendSessionMessage(session.session_id, question, language, location);
      setMessages((items) => [...items, responseMessage(response)]);
      setCitations(response.sources);
      setMetadata(response.metadata);
      setWarnings(response.warnings);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Message failed.");
    } finally {
      setSending(false);
    }
  }

  function shareLocation() {
    if (!navigator.geolocation) {
      setLocationNote("Location is not available in this browser.");
      return;
    }
    setLocationNote("Getting location…");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          latitude: Math.round(position.coords.latitude * 100) / 100,
          longitude: Math.round(position.coords.longitude * 100) / 100,
        });
        setLocationNote("Location shared for the weather check (rounded to ~1 km).");
      },
      () => setLocationNote("Location permission was denied; weather will be skipped."),
      { timeout: 10000 },
    );
  }

  async function reassessWithPhoto(file: File) {
    if (!session || sending) return;
    const now = new Date().toISOString();
    setMessages((items) => [...items, {
      message_id: `pending-${now}`,
      role: "user",
      content: `New photo uploaded: ${file.name}. Has it improved?`,
      citations: [],
      metadata: {},
      created_at: now,
    }]);
    setSending(true);
    setError(null);
    try {
      const diagnosis = await diagnoseImage(file);
      const response = await reassessSession(session.session_id, diagnosis.session_id, location);
      setMessages((items) => [...items, responseMessage(response)]);
      setCitations(response.sources);
      setMetadata(response.metadata);
      setWarnings(response.warnings);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Reassessment failed.");
    } finally {
      setSending(false);
      if (photoRef.current) photoRef.current.value = "";
    }
  }

  function pickQuestion(question: string) {
    setDraft(`${question} `);
    inputRef.current?.focus();
  }

  async function resetChat() {
    if (session) await deleteChatSession(session.session_id).catch(() => undefined);
    setLoading(true);
    setMessages([]);
    setCitations([]);
    setMetadata(null);
    setWarnings([]);
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
      <div className="ask-layout">
        <section className="panel ask-conversation">
          <div className="chat-status">
            <h1 className="sr-only">Ask CropAI</h1>
            <span className="status-dot" /> <strong>Ask CropAI · {modeLabel}</strong><small>{session ? `Session ${session.session_id.slice(0, 8)}` : "Starting session…"}</small>
            <label className="sr-only" htmlFor="ask-language">Language</label>
            <select id="ask-language" value={language} onChange={(event) => setLanguage(event.target.value as "en" | "kn" | "hi")}><option value="en">English</option><option value="kn">ಕನ್ನಡ</option><option value="hi">हिन्दी</option></select>
            <button className="button secondary" type="button" onClick={resetChat}>New chat</button>
          </div>
          {session?.mode === "diagnosis" && <div className="callout">Visual evidence from diagnosis {session.diagnosis_id?.slice(0, 8)} is attached. A team of CropAI agents diagnoses, checks trusted fact sheets, assesses risk and plans treatment and follow-up.</div>}
          {session?.mode === "diagnosis" && <div className="agent-toolbar"><button type="button" className="button secondary" onClick={shareLocation} disabled={Boolean(location)}>{location ? "Location shared ✓" : "Share location for weather"}</button><button type="button" className="button secondary" onClick={() => photoRef.current?.click()} disabled={sending}>Reassess with new photo</button><input ref={photoRef} type="file" accept="image/*" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void reassessWithPhoto(file); }} />{locationNote && <small className="muted">{locationNote}</small>}</div>}
          <div className="conversation ask-thread" aria-live="polite" ref={threadRef}>
            {!loading && messages.length === 0 && <div className="empty-chat"><strong>Ask a focused crop-health question.</strong><p>Include the crop, visible symptoms, location context, and uncertainty. Do not include secrets or personal information.</p><div className="suggestion-row">{starterQuestions.map((question) => <button key={question} type="button" disabled={!session || sending} onClick={() => submit(undefined, question)}>{question}</button>)}</div></div>}
            {messages.map((item) => <article className={`message ${item.role}`} key={item.message_id}><div className="message-heading"><strong>{item.role === "user" ? "You" : "CropAI"}</strong><time>{new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div><p>{item.content}</p>{item.role === "assistant" && <AgentInsights metadata={item.metadata} onPickQuestion={pickQuestion} />}{item.citations.length > 0 && <button type="button" className="citation-link" onClick={() => setCitations(item.citations)}>{item.citations.length} cited passage{item.citations.length === 1 ? "" : "s"}</button>}{item.role === "assistant" && <button type="button" className="copy-link" onClick={() => navigator.clipboard.writeText(item.content)}>Copy</button>}</article>)}
            {sending && <div className="message assistant loading-message">{session?.mode === "diagnosis" ? "CropAI agents are working: checking evidence, diagnosing, assessing risk and planning…" : "Retrieving evidence and preparing a cautious response…"}</div>}
          </div>
          <form className="ask-form" onSubmit={submit}><label className="sr-only" htmlFor="ask-message">Question</label><textarea id="ask-message" ref={inputRef} rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (draft.trim() && !sending && session) void submit(); } }} maxLength={2000} placeholder="Ask about crop symptoms, uncertainty, prevention evidence… (Enter to send, Shift+Enter for a new line)" /><button className="button primary" disabled={!draft.trim() || sending || !session}>Send</button>{draft.length > 1500 && <small className="char-count">{draft.length}/2000</small>}</form>
          {error && <div className="chat-error" role="alert">{error}{lastQuestion && <button type="button" onClick={() => submit(undefined, lastQuestion)}>Retry</button>}</div>}
          <p className="muted chat-disclaimer">Decision support only. Verify important actions with a qualified agricultural expert.</p>
        </section>

        <aside className="source-drawer">
          <div className="panel"><p className="eyebrow">Grounding</p><h2>Sources</h2>{citations.length === 0 ? <p className="muted">No source is attached to the selected answer. Treat it as unsupported or insufficient evidence.</p> : citations.map((source) => <article className="source-card" key={source.source_id}><span>{source.source_id}</span><h3>{source.title}</h3><p>{source.excerpt}</p><small>{[source.organization, source.section_name, source.page ? `page ${source.page}` : null].filter(Boolean).join(" · ")}</small>{source.url && <a href={source.url} target="_blank" rel="noreferrer">Open source ↗</a>}</article>)}</div>
          <div className="panel pipeline-meta"><p className="eyebrow">Pipeline metadata</p><dl><div><dt>Mode</dt><dd>{modeLabel}</dd></div><div><dt>Provider</dt><dd>{metadata?.provider ?? session?.provider ?? "—"}</dd></div><div><dt>Model</dt><dd>{metadata?.model || session?.model || "—"}</dd></div><div><dt>Retriever</dt><dd>{metadata?.retriever ?? "—"}</dd></div><div><dt>Retrieved</dt><dd>{metadata?.retrieved_chunks ?? 0} chunks</dd></div><div><dt>Total time</dt><dd>{metadata ? `${metadata.total_ms.toFixed(0)} ms` : "—"}</dd></div></dl>{latestAssistant?.citations.length === 0 && <span className="tag caution-tag">No citation</span>}{warnings.length > 0 && <ul className="pipeline-warnings">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}</div>
        </aside>
      </div>
    </main>
  );
}
