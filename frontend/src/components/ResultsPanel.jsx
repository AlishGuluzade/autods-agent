import React from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ResultsPanel({ status, jobId, backendBase }) {
  if (!status || status.status !== "done") return null;

  const { metrics, best_model_name, explanation_text, top_features } = status;
  const chartData = [...(top_features || [])].reverse();

  return (
    <div className="panel">
      <h2>3. Results</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Best model</div>
          <div className="metric-value" style={{ fontSize: 18 }}>
            {best_model_name}
          </div>
        </div>
        {Object.entries(metrics || {}).map(([key, value]) => (
          <div className="metric-card" key={key}>
            <div className="metric-label">{key.replace("_", " ")}</div>
            <div className="metric-value">{value.toFixed(3)}</div>
          </div>
        ))}
      </div>

      {chartData.length > 0 && (
        <div style={{ height: 260, marginBottom: 22 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="importance" fill="#0f6e5e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="explanation-text">{explanation_text}</div>

      <a className="report-btn" href={`${backendBase}/report/${jobId}`} target="_blank" rel="noreferrer">
        Download PDF report
      </a>
    </div>
  );
}
