// Abstract "finger extension" bars per intent (100 = fully extended, 0 = fully curled).
// This is a schematic representation, not a literal hand render — it keeps the
// dashboard focused on signal -> intent -> reasoning rather than illustration.
const FINGER_PROFILES = {
  grasp: [20, 15, 15, 15, 20],
  release: [90, 90, 90, 90, 90],
  hold_firm: [10, 10, 10, 10, 12],
  gentle_hold: [45, 38, 38, 38, 45],
  pinch: [30, 30, 80, 82, 80],
};

const LABELS = {
  grasp: "Grasp",
  release: "Release",
  hold_firm: "Hold — Firm",
  gentle_hold: "Hold — Gentle",
  pinch: "Pinch",
};

export default function GripDisplay({ intent, confidence }) {
  const profile = FINGER_PROFILES[intent] || FINGER_PROFILES.release;

  return (
    <div className="panel grip-panel">
      <div className="panel-label">
        <span className="eyebrow">02 — PREDICTED INTENT</span>
      </div>
      <div className="grip-body">
        <div className="finger-bars" aria-hidden="true">
          {profile.map((h, idx) => (
            <div className="finger-track" key={idx}>
              <div className="finger-fill" style={{ height: `${h}%` }} />
            </div>
          ))}
        </div>
        <div className="grip-meta">
          <div className="grip-name">{LABELS[intent] || intent}</div>
          <div className="grip-confidence">
            <div className="grip-confidence-bar">
              <div
                className="grip-confidence-fill"
                style={{ width: `${Math.round(confidence * 100)}%` }}
              />
            </div>
            <span className="mono">{Math.round(confidence * 100)}% confidence</span>
          </div>
        </div>
      </div>
    </div>
  );
}
