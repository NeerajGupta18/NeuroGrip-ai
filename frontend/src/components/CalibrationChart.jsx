import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from "recharts";

export default function CalibrationChart({ profile, evaluation, grasps }) {
  if (!profile) return <div className="panel" />;
  const history = profile.rolling_accuracy || [];
  const data = history.map((v, i) => ({ n: i + 1, accuracy: v }));
  const learned = profile.recordings_learned || {};
  const minReps = grasps.length ? Math.min(...grasps.map((g) => learned[g.key] || 0)) : 0;
  const curve = evaluation?.calibration_curve?.reps_per_grasp || {};
  const steps = Object.keys(curve).map(Number).sort((a, b) => a - b);
  const expectedKey = steps.filter((s) => s <= minReps).pop();
  const expected = expectedKey !== undefined ? curve[String(expectedKey)].recording : null;

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">04 — CALIBRATION</span>
        <span className="panel-sub">
          {profile.calibrated_grasps}/{grasps.length} grasps calibrated · {profile.recordings_confirmed} recordings confirmed
        </span>
      </div>

      <div className="calib-chips">
        {grasps.map((g) => (
          <span key={g.key} className={`calib-chip mono ${learned[g.key] ? "calib-chip-on" : ""}`}>
            {g.label} ×{learned[g.key] || 0}
          </span>
        ))}
      </div>

      {data.length > 1 ? (
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -20 }}>
            <XAxis dataKey="n" tick={{ fill: "#8a9499", fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "#8a9499", fontSize: 11 }} />
            {expected !== null && <ReferenceLine y={expected} stroke="#4fd1c5" strokeDasharray="4 4" />}
            <Tooltip
              contentStyle={{
                background: "#1a2226",
                border: "1px solid #384349",
                borderRadius: 6,
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: 12,
              }}
              formatter={(value) => [`${value}%`, `accuracy, last ${profile.rolling_window}`]}
              labelFormatter={(l) => `recording ${l}`}
            />
            <Line type="monotone" dataKey="accuracy" stroke="#f0a868" strokeWidth={2} dot={{ r: 2, fill: "#f0a868" }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="empty-calibration">
          Nothing confirmed yet. Press &ldquo;Calibrate&rdquo; below: each recording is predicted first, scored, and only
          then learned from.
        </div>
      )}

      <div className="calib-stats mono">
        <span>
          rolling accuracy (last {profile.rolling_window}): <strong>{profile.recent_accuracy ?? "–"}%</strong>
        </span>
        <span>
          when acting: <strong>{profile.accuracy_when_acting ?? "–"}%</strong> · abstained {profile.abstained}
        </span>
        {expected !== null && (
          <span className="calib-expected">
            dashed line: {expected}% expected by the offline evaluation at {expectedKey} recording
            {expectedKey === 1 ? "" : "s"} per grasp
          </span>
        )}
      </div>
    </div>
  );
}
