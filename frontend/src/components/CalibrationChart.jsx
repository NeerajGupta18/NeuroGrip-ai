import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from "recharts";

export default function CalibrationChart({ accuracyHistory, sessionCount }) {
  const data = accuracyHistory.map((v, i) => ({ session: i + 1, accuracy: v }));

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">04 — ADAPTIVE CALIBRATION</span>
        <span className="panel-sub">
          {sessionCount === 0
            ? "no feedback yet — confirm a prediction below to begin calibrating"
            : `${sessionCount} confirmed session${sessionCount === 1 ? "" : "s"} for this user profile`}
        </span>
      </div>
      {accuracyHistory.length > 1 ? (
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -20 }}>
            <XAxis dataKey="session" tick={{ fill: "#8a9499", fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "#8a9499", fontSize: 11 }} />
            <ReferenceLine y={100} stroke="#2a3338" />
            <Tooltip
              contentStyle={{
                background: "#1a2226",
                border: "1px solid #384349",
                borderRadius: 6,
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: 12,
              }}
              formatter={(value) => [`${value}%`, "running accuracy"]}
              labelFormatter={(l) => `session ${l}`}
            />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#f0a868"
              strokeWidth={2}
              dot={{ r: 3, fill: "#f0a868" }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="empty-calibration">Confirm at least 2 predictions to see a trend.</div>
      )}
    </div>
  );
}
