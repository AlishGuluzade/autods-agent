import React, { useEffect, useRef, useState } from "react";
import UploadCard from "./components/UploadCard.jsx";
import TracePanel from "./components/TracePanel.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

// All requests go through /api, which vite.config.js proxies to the FastAPI
// backend in dev, and which your deployment's reverse proxy (nginx, HF
// Spaces, etc.) should do the same for in production.
const API_BASE = "/api";
const POLL_INTERVAL_MS = 1500;

export default function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [errorDetail, setErrorDetail] = useState(null);
  const [isBusy, setIsBusy] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const startPolling = (id) => {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${id}`);
        if (!res.ok) throw new Error(`Status check failed (${res.status})`);
        const data = await res.json();
        setStatus(data);

        if (data.status === "done" || data.status === "error") {
          clearInterval(pollRef.current);
          setIsBusy(false);
          if (data.status === "error") {
            setError(data.error || "The agent hit an error processing this dataset.");
            setErrorDetail(data.error_detail || null);
          }
        }
      } catch (err) {
        clearInterval(pollRef.current);
        setIsBusy(false);
        setError(err.message);
      }
    }, POLL_INTERVAL_MS);
  };

  const handleAnalyze = async () => {
    setError(null);
    setErrorDetail(null);
    setStatus(null);
    setIsBusy(true);

    try {
      const form = new FormData();
      form.append("file", file);
      const uploadRes = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
      if (!uploadRes.ok) throw new Error("Upload failed. Is the backend running?");
      const { dataset_id } = await uploadRes.json();

      const analyzeRes = await fetch(`${API_BASE}/analyze/${dataset_id}`, { method: "POST" });
      if (!analyzeRes.ok) throw new Error("Could not start the analysis.");
      const { job_id } = await analyzeRes.json();

      setJobId(job_id);
      startPolling(job_id);
    } catch (err) {
      setIsBusy(false);
      setError(err.message);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <p className="eyebrow">AutoDS Agent</p>
        <h1>Hand it a spreadsheet. It hands back a model.</h1>
        <p>
          Upload a CSV and the agent profiles it, plans a pipeline, engineers features,
          tunes several models with Optuna, explains what it found with SHAP, and writes
          up a PDF report — with no manual steps in between.
        </p>
      </header>

      {error && (
        <div className="error-banner">
          <strong>Something went wrong.</strong> {error}
          {errorDetail && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: "pointer" }}>Technical detail</summary>
              <code style={{ fontSize: 12 }}>{errorDetail}</code>
            </details>
          )}
        </div>
      )}

      <UploadCard file={file} onFileSelected={setFile} onAnalyze={handleAnalyze} isBusy={isBusy} />

      {(status || isBusy) && <TracePanel logs={status?.logs || []} currentStatus={status?.status || "profile"} />}

      {status && jobId && <ResultsPanel status={status} jobId={jobId} backendBase={API_BASE} />}

      <p className="footer-note">github.com/AlishGuluzade/autods-agent</p>
    </div>
  );
}