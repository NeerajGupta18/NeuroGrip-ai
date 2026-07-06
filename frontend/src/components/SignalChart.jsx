import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

export default function SignalChart({ signal }) {
  const data = signal.map((v, i) => ({ i, v }));

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">01 — RAW SIGNAL</span>
        <span className="panel-sub">simulated myoelectric window · {signal.length} samples</span>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -20 }}>
          <XAxis dataKey="i" hide />
          <YAxis domain={[-1.2, 1.2]} hide />
          <Tooltip
            contentStyle={{
              background: "#1a2226",
              border: "1px solid #384349",
              borderRadius: 6,
              fontFamily: "IBM Plex Mono, monospace",
              fontSize: 12,
            }}
            labelFormatter={() => ""}
            formatter={(value) => [value.toFixed(3), "amplitude"]}
          />
          <Line type="monotone" dataKey="v" stroke="#4fd1c5" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
