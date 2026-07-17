import React from "react";

// Fixed step order so the timeline renders consistently even if a step
// hasn't logged a message yet (it just shows as pending).
const STEP_ORDER = ["profile", "plan", "feature_eng", "train", "explain", "report", "done"];

const STEP_LABELS = {
  profile: "Profiling data",
  plan: "Planning pipeline",
  feature_eng: "Engineering features",
  train: "Training & tuning models",
  explain: "Explaining results",
  report: "Generating report",
  done: "Complete",
};

export default function TracePanel({ logs, currentStatus }) {
  // Group log messages by step so each timeline entry can show its latest message.
  const messagesByStep = {};
  for (const entry of logs) {
    messagesByStep[entry.step] = entry.message;
  }

  const currentIndex = STEP_ORDER.indexOf(currentStatus);

  return (
    <div className="panel">
      <h2>2. Agent trace</h2>
      <ul className="trace">
        {STEP_ORDER.map((step, i) => {
          const state = i < currentIndex || currentStatus === "done" ? "complete" : i === currentIndex ? "active" : "";
          return (
            <li key={step} className={`trace-item ${state}`}>
              <span className="trace-dot" />
              <div className="trace-step">{STEP_LABELS[step]}</div>
              <div className="trace-message">
                {messagesByStep[step] || (state === "" ? "Waiting…" : "")}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
