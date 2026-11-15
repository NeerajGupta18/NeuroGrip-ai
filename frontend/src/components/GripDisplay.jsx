// Schematic finger-extension bars per grasp (100 = fully extended, 0 = fully curled),
// thumb first. A diagram of the hand posture, not a literal render.
const FINGER_PROFILES = {
  cyl: [30, 20, 20, 20, 20],
  hook: [85, 25, 25, 25, 25],
  tip: [45, 45, 85, 85, 85],
  palm: [40, 55, 55, 55, 55],
  spher: [50, 45, 45, 45, 45],
  lat: [35, 60, 30, 30, 30],
};
const HOLD = [50, 50, 50, 50, 50];

const ABSTAIN_TEXT = {
  sensor_fault: "Sensor fault",
  not_calibrated: "Wearer not calibrated",
  low_confidence: "Confidence below floor",
};

export default function GripDisplay({ prediction, graspLabels }) {
  const acting = prediction.decision === "act";
  const grasp = prediction.predicted_grasp;
  const confidence = prediction.confidence ?? 0;
  const profile = acting ? FINGER_PROFILES[grasp] || HOLD : HOLD;
  const votes = prediction.window_votes || [];

  return (
    <div className={`panel grip-panel ${acting ? "" : "grip-panel-abstain"}`}>
      <div className="panel-label">
        <span className="eyebrow">02 — DECISION</span>
        <span className={`decision-pill ${acting ? "decision-act" : "decision-abstain"}`}>
          {acting ? "ACT" : "ABSTAIN"}
        </span>
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
          <div className="grip-name">{acting ? graspLabels[grasp] || grasp : "Hold last safe grip"}</div>
          {!acting && (
            <div className="abstain-reason mono">
              {ABSTAIN_TEXT[prediction.abstain_reason]}
              {grasp ? ` · best guess ${graspLabels[grasp] || grasp}` : ""}
            </div>
          )}
          {grasp && (
            <div className="grip-confidence">
              <div className="grip-confidence-bar">
                <div className="grip-confidence-fill" style={{ width: `${Math.round(confidence * 100)}%` }} />
                <div
                  className="confidence-floor-mark"
                  style={{ left: `${Math.round(prediction.confidence_floor * 100)}%` }}
                  title="confidence floor required to act"
                />
              </div>
              <span className="mono">
                {Math.round(confidence * 100)}% confidence · floor {Math.round(prediction.confidence_floor * 100)}%
              </span>
            </div>
          )}
        </div>
      </div>
      {votes.length > 0 && (
        <div className="vote-strip-wrap">
          <span className="panel-sub">each half-second window&rsquo;s own vote, in time order</span>
          <div className="vote-strip">
            {votes.map((v, i) => (
              <span
                key={i}
                className={`vote-cell ${v === grasp ? "vote-agree" : "vote-differ"}`}
                title={`${(i * 0.25).toFixed(2)} s: ${graspLabels[v] || v}`}
              />
            ))}
          </div>
        </div>
      )}
      <div className="panel-sub mono latency">decided in {prediction.latency_ms} ms</div>
    </div>
  );
}
