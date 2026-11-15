export default function EvaluationPanel({ evaluation }) {
  const curve = evaluation.calibration_curve.reps_per_grasp;
  const safety = evaluation.safety_gate;
  const tiles = [
    {
      value: `${evaluation.within_wearer.recording}%`,
      label: "trained on the wearer",
      note: "5-fold over recordings",
    },
    {
      value: `${evaluation.unseen_wearer_no_calibration.recording}%`,
      label: "new wearer, no calibration",
      note: `chance is ${evaluation.dataset.chance_accuracy}%`,
    },
    { value: `${curve["1"].recording}%`, label: "after 1 recording per grasp", note: "about 36 s of data" },
    { value: `${curve["3"].recording}%`, label: "after 3 recordings per grasp", note: "under 2 minutes" },
    {
      value: `${safety.false_alarms}/${safety.clean_recordings}`,
      label: "safety-gate false alarms",
      note: "on clean recordings",
    },
    {
      value: `${evaluation.latency_ms_per_recording} ms`,
      label: "per decision",
      note: "single CPU core",
    },
  ];

  return (
    <section className="panel eval-panel">
      <div className="panel-label">
        <span className="eyebrow">MEASURED RESULTS</span>
        <span className="panel-sub">
          whole-recording accuracy, leave-one-wearer-out · {evaluation.dataset.wearers} wearers ·{" "}
          {evaluation.dataset.grasps} grasps · full write-up in docs/EVALUATION.md
        </span>
      </div>
      <div className="eval-grid">
        {tiles.map((t) => (
          <div className="eval-tile" key={t.label}>
            <span className="eval-value">{t.value}</span>
            <span className="eval-label">{t.label}</span>
            <span className="eval-note mono">{t.note}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
