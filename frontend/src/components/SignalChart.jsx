import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

const MAX_POINTS = 750; // decimate for drawing only; the model sees every sample

export default function SignalChart({ recording }) {
  const { ch1, ch2, sample_rate: rate } = recording;
  const stride = Math.max(1, Math.ceil(ch1.length / MAX_POINTS));
  const data = [];
  for (let i = 0; i < ch1.length; i += stride) {
    data.push({ t: +(i / rate).toFixed(3), ch1: ch1[i], ch2: ch2[i] });
  }
  const tooltipStyle = {
    background: "#1a2226",
    border: "1px solid #384349",
    borderRadius: 6,
    fontFamily: "IBM Plex Mono, monospace",
    fontSize: 12,
  };

  return (
    <div className="panel">
      <div className="panel-label">
        <span className="eyebrow">01 — RAW SIGNAL</span>
        <span className="panel-sub">
          real surface EMG · 2 channels · {(ch1.length / rate).toFixed(0)} s at {rate} Hz
        </span>
      </div>
      {[
        ["ch1", "#4fd1c5", "channel 1"],
        ["ch2", "#f0a868", "channel 2"],
      ].map(([key, color, label]) => (
        <div key={key} className="signal-row">
          <span className="signal-row-label mono" style={{ color }}>
            {label}
          </span>
          <ResponsiveContainer width="100%" height={96}>
            <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -20 }}>
              <XAxis dataKey="t" hide />
              <YAxis domain={["auto", "auto"]} hide />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(t) => `${t} s`}
                formatter={(value) => [Number(value).toFixed(3), label]}
              />
              <Line type="linear" dataKey={key} stroke={color} strokeWidth={1} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
