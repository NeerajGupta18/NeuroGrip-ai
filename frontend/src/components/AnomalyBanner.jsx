export default function AnomalyBanner({ anomalies }) {
  if (!anomalies || anomalies.length === 0) return null;

  const highestSeverity = anomalies.some((a) => a.severity === "high") ? "high" : "medium";

  return (
    <div className={`anomaly-banner anomaly-${highestSeverity}`}>
      <div className="anomaly-header">
        <span className="anomaly-icon" aria-hidden="true">
          ⚠
        </span>
        <span className="anomaly-title">
          Signal quality issue{anomalies.length > 1 ? "s" : ""} detected — prediction is provisional
        </span>
      </div>
      <ul className="anomaly-list">
        {anomalies.map((a, i) => (
          <li key={i} className={`anomaly-item anomaly-item-${a.severity}`}>
            <span className="anomaly-type mono">{a.type}</span>
            <span className="anomaly-message">{a.message}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
