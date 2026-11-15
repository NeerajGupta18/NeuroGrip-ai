export default function AnomalyBanner({ anomalies }) {
  if (!anomalies || anomalies.length === 0) return null;

  return (
    <div className="anomaly-banner anomaly-high" role="alert">
      <div className="anomaly-header">
        <span className="anomaly-icon" aria-hidden="true">
          ⚠
        </span>
        <span className="anomaly-title">
          Sensor fault detected — classification blocked, hand holds its last safe grip
        </span>
      </div>
      <ul className="anomaly-list">
        {anomalies.map((a, i) => (
          <li key={i} className={`anomaly-item anomaly-item-${a.severity}`}>
            <span className="anomaly-type mono">
              {a.type} · {a.channel}
            </span>
            <span className="anomaly-message">{a.message}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
