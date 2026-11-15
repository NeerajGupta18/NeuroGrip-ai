import { useState } from "react";
import { getReport } from "../api.js";

function formatReport(report, graspLabels, sourceLabels) {
  const pctOrDash = (v) => (v === null || v === undefined ? "-" : `${v}%`);
  const lines = [];
  lines.push("NEUROGRIP AI - SESSION REPORT");
  lines.push("=".repeat(46));
  lines.push(`Profile:               ${report.user_id}`);
  lines.push(`Signal source:         ${sourceLabels[report.source] || report.source}`);
  lines.push(`Generated:             ${new Date(report.generated_at).toLocaleString()}`);
  lines.push("");
  lines.push("CALIBRATION");
  lines.push("-".repeat(46));
  lines.push(`Grasps calibrated:     ${report.calibrated_grasps} of ${Object.keys(report.recordings_learned).length}`);
  Object.entries(report.recordings_learned).forEach(([g, n]) => {
    lines.push(`  ${(graspLabels[g] || g).padEnd(14)} ${n} recording${n === 1 ? "" : "s"} learned`);
  });
  lines.push("");
  lines.push("ACCURACY (each recording predicted before it was learned from)");
  lines.push("-".repeat(46));
  lines.push(`Recordings confirmed:  ${report.recordings_confirmed}`);
  lines.push(`Overall accuracy:      ${pctOrDash(report.overall_accuracy)}`);
  lines.push(`Last ${report.rolling_window} recordings:    ${pctOrDash(report.recent_accuracy)}`);
  lines.push(`When the agent acted:  ${pctOrDash(report.accuracy_when_acting)}`);
  lines.push(`Abstained:             ${report.abstained}`);
  lines.push("");
  lines.push("EXPECTED (offline evaluation, recordings per grasp -> accuracy)");
  lines.push("-".repeat(46));
  Object.entries(report.expected_accuracy_by_reps).forEach(([reps, acc]) => {
    lines.push(`  ${String(reps).padStart(2)} -> ${acc}%`);
  });
  lines.push("");
  lines.push("Recordings replayed from the UCI 'sEMG for Basic Hand movements'");
  lines.push("dataset (CC BY 4.0). Research prototype, not a medical device.");
  return lines.join("\n");
}

export default function ReportExport({ userId, graspLabels, sourceLabels }) {
  const [status, setStatus] = useState("idle"); // idle | loading | error

  const handleExport = async () => {
    setStatus("loading");
    try {
      const report = await getReport(userId);
      const blob = new Blob([formatReport(report, graspLabels, sourceLabels)], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neurogrip_session_report_${userId}.txt`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  };

  return (
    <button className="btn btn-ghost" onClick={handleExport} disabled={status === "loading"}>
      {status === "loading" ? "Preparing report…" : "Export session report"}
      {status === "error" && <span className="export-error"> — failed, try again</span>}
    </button>
  );
}
