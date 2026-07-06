const LABELS = {
  grasp: "Grasp",
  release: "Release",
  hold_firm: "Hold-Firm",
  gentle_hold: "Hold-Gentle",
  pinch: "Pinch",
};

export default function ReasoningPanel({ reasoning, confidenceByIntent, features }) {
  const sorted = Object.entries(confidenceByIntent || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">03 — AGENT REASONING</span>
        <span className="panel-sub">why the model made this call</span>
      </div>

      <p className="reasoning-text">{reasoning}</p>

      <div className="confidence-breakdown">
        {sorted.map(([intent, conf]) => (
          <div className="confidence-row" key={intent}>
            <span className="confidence-row-label mono">{LABELS[intent] || intent}</span>
            <div className="confidence-row-track">
              <div className="confidence-row-fill" style={{ width: `${conf * 100}%` }} />
            </div>
            <span className="confidence-row-value mono">{Math.round(conf * 100)}%</span>
          </div>
        ))}
      </div>

      {features && (
        <div className="feature-grid mono">
          {Object.entries(features).map(([k, v]) => (
            <div className="feature-cell" key={k}>
              <span className="feature-key">{k.toUpperCase()}</span>
              <span className="feature-value">{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
