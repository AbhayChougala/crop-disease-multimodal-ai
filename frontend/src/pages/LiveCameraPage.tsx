import { useEffect, useRef, useState } from "react";

import { diagnoseImage, WS_URL } from "../api/client";
import { DiagnosisResults } from "../components/DiagnosisResults";
import type { DiagnosisResponse, LivePrediction } from "../types/api";

export function LiveCameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const inFlight = useRef(false);
  const [running, setRunning] = useState(false);
  const [prediction, setPrediction] = useState<LivePrediction | null>(null);
  const [fullResult, setFullResult] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);

  function stopCamera() {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    socketRef.current?.close(); socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    inFlight.current = false; setRunning(false);
  }

  useEffect(() => stopCamera, []);

  function frameBlob(): Promise<Blob | null> {
    const video = videoRef.current; const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return Promise.resolve(null);
    const width = Math.min(720, video.videoWidth);
    const height = Math.round(width * video.videoHeight / video.videoWidth);
    canvas.width = width; canvas.height = height;
    canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
  }

  async function startCamera() {
    setError(null); setFullResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      const socket = new WebSocket(WS_URL); socket.binaryType = "arraybuffer"; socketRef.current = socket;
      socket.onmessage = (event) => { inFlight.current = false; const data = JSON.parse(event.data) as LivePrediction; data.error ? setError(data.error) : setPrediction(data); };
      socket.onerror = () => { inFlight.current = false; setError("Realtime connection failed. Is the backend running?"); };
      socket.onclose = () => { inFlight.current = false; };
      socket.onopen = () => {
        timerRef.current = window.setInterval(async () => {
          if (socket.readyState !== WebSocket.OPEN || inFlight.current) return;
          const blob = await frameBlob(); if (!blob) return;
          inFlight.current = true; socket.send(await blob.arrayBuffer());
        }, 500);
      };
      setRunning(true);
    } catch (problem) {
      stopCamera(); setError(problem instanceof Error ? problem.message : "Camera permission was not granted.");
    }
  }

  async function captureDiagnosis() {
    setCapturing(true); setError(null);
    try { const blob = await frameBlob(); if (!blob) throw new Error("No camera frame is available."); setFullResult(await diagnoseImage(blob)); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Capture failed."); }
    finally { setCapturing(false); }
  }

  const fps = prediction?.total_ms ? Math.min(2, 1000 / Math.max(500, prediction.total_ms)).toFixed(1) : "—";
  return (
    <main className="page-shell">
      <section className="page-intro"><p className="eyebrow">Realtime workflow</p><h1>Live camera analysis</h1><p>Frames are sampled twice per second. Only one frame is processed at a time, preventing a stale-frame queue.</p></section>
      <section className="camera-layout">
        <div className="camera-stage">
          <video ref={videoRef} playsInline muted />
          {!running && <div className="camera-placeholder">Camera preview</div>}
          {prediction && running && <div className="camera-overlay"><span>{prediction.is_mock ? "DEMO MODE" : "REAL MODEL"}</span><strong>{prediction.crop} · {prediction.disease}</strong><small>{(prediction.confidence * 100).toFixed(1)}% {prediction.is_mock ? "placeholder confidence" : "confidence"}</small></div>}
        </div>
        <aside className="camera-controls">
          <span className="status-pill">2 FPS target</span><h2>Live telemetry</h2>
          <dl><div><dt>Total latency</dt><dd>{prediction ? `${prediction.total_ms.toFixed(1)} ms` : "—"}</dd></div><div><dt>Inference</dt><dd>{prediction ? `${prediction.inference_ms.toFixed(1)} ms` : "—"}</dd></div><div><dt>Effective FPS</dt><dd>{fps}</dd></div></dl>
          <div className="button-stack">{running ? <button className="button secondary" onClick={stopCamera}>Stop camera</button> : <button className="button primary" onClick={startCamera}>Start camera</button>}<button className="button primary" disabled={!running || capturing} onClick={captureDiagnosis}>{capturing ? "Capturing…" : "Capture & full diagnosis"}</button></div>
          {error && <p className="error" role="alert">{error}</p>}
        </aside>
      </section>
      <canvas ref={canvasRef} hidden />
      {fullResult && <DiagnosisResults result={fullResult} />}
    </main>
  );
}
