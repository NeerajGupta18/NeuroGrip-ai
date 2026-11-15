export default function ReasoningPanel({ prediction, graspLabels }) {
  const sorted = Object.entries(prediction.confidence_by_grasp || {}).sort((a, b) => b[1] - a[1]);
  const contributions = (prediction.contributions || []).slice(0, 6);
  const largest = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 1e-9);
  const winner = graspLabels[prediction.predicted_grasp];
  const rival = graspLabels[prediction.runner_up];

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">03 — WHY</span>
        <span className="panel-sub">
          {prediction.reasoning_source === "llm"
            ? "computed evidence, re-worded by an LLM"
            : "computed directly from the model"}
        </span>
      </div>

      <p className="reasoning-text">{prediction.reasoning}</p>

      {sorted.length > 0 && (
        <div className="confidence-breakdown">
          {sorted.map(([grasp, conf]) => (
            <div className="confidence-row" key={grasp}>
              <span className="confidence-row-label mono">{graspLabels[grasp] || grasp}</span>
              <div className="confidence-row-track">
                <div className="confidence-row-fill" style={{ width: `${conf * 100}%` }} />
              </div>
              <span className="confidence-row-value mono">{Math.round(conf * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {contributions.length > 0 && (
        <div className="contrib-block">
          <span className="panel-sub">
            evidence for {winner} (right, teal) vs {rival} (left, amber) — six strongest features
          </span>
          {contributions.map((c) => {
            const width = (50 * Math.abs(c.contribution)) / largest;
            const positive = c.contribution > 0;
            return (
              <div className="contrib-row" key={c.feature}>
                <span className="contrib-label mono">
                  {c.label} · {c.channel}
                </span>
                <div className="contrib-track">
                  <div
                    className={`contrib-fill ${positive ? "contrib-for" : "contrib-against"}`}
                    style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                  />
                  <div className="contrib-zero" />
                </div>
                <span className="contrib-value mono">
                  {positive ? "+" : ""}
                  {c.contribution.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
