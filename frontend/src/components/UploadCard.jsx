import React, { useRef, useState } from "react";

const EXAMPLE_HINT = "No CSV handy? Try the churn_sample.csv from the examples/ folder in this repo.";

export default function UploadCard({ file, onFileSelected, onAnalyze, isBusy }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (fileList) => {
    const picked = fileList?.[0];
    if (picked && picked.name.endsWith(".csv")) {
      onFileSelected(picked);
    }
  };

  return (
    <div className="panel">
      <h2>1. Upload a dataset</h2>

      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="dropzone-label">
          <strong>Click to choose a CSV</strong>, or drag one here.
        </div>
        {file && <div className="file-chip">{file.name}</div>}
      </div>

      <p style={{ fontSize: 13, color: "var(--ink-soft)", marginTop: 12 }}>{EXAMPLE_HINT}</p>

      <button className="primary-btn" disabled={!file || isBusy} onClick={onAnalyze}>
        {isBusy ? "Agent is working…" : "Run the agent"}
      </button>
    </div>
  );
}
