import { useState } from "react";
import { getReport } from "../api.js";

const INTENT_LABELS = {
  grasp: "Grasp",
  release: "Release",
  hold_firm: "Hold-Firm",
  gentle_hold: "Hold-Gentle",
  pinch: "Pinch",
};

function formatReport(report) {
  const lines = [];
  lines.push("NEUROGRIP AI — SESSION REPORT");
  lines.push("=".repeat(40));
  lines.push(`User:            ${report.user_id}`);
  lines.push(`Generated:       ${new Date(report.generated_at).toLocaleString()}`);
  lines.push("");
  lines.push("CALIBRATION SUMMARY");
  lines.push("-".repeat(40));
  lines.push(`Confirmed sessions:    ${report.session_count}`);
  lines.push(`Overall accuracy:      ${report.overall_accuracy}%`);
  lines.push("");
  lines.push("Accuracy trend (running, per confirmed session):");
  lines.push(`  ${report.accuracy_history.join("% -> ")}${report.accuracy_history.length ? "%" : "(no confirmed sessions yet)"}`);
  lines.push("");
  lines.push("GRIP INTENT DISTRIBUTION (confirmed sessions)");
  lines.push("-".repeat(40));
  Object.entries(report.intent_distribution).forEach(([intent, count]) => {
    lines.push(`  ${(INTENT_LABELS[intent] || intent).padEnd(14)} ${count}`);
  });
  lines.push("");
  lines.push(
    "Note: generated from a simulated-signal prototype, not a hardware-connected"
  );
  lines.push("device. Intended to demonstrate the reporting layer a clinician or");
  lines.push("caregiver would use to review a remote rehabilitation session.");
  return lines.join("\n");
}

export default function ReportExport({ userId }) {
  const [status, setStatus] = useState("idle"); // idle | loading | error

  const handleExport = async () => {
    setStatus("loading");
    try {
      const report = await getReport(userId);
      const text = formatReport(report);
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neurogrip_session_report_${userId}.txt`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("idle");
    } catch (err) {
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
