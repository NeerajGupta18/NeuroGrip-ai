import { useCallback, useEffect, useMemo, useState } from "react";
import SignalChart from "./components/SignalChart.jsx";
import GripDisplay from "./components/GripDisplay.jsx";
import ReasoningPanel from "./components/ReasoningPanel.jsx";
import CalibrationChart from "./components/CalibrationChart.jsx";
import AnomalyBanner from "./components/AnomalyBanner.jsx";
import UserSwitcher from "./components/UserSwitcher.jsx";
import ReportExport from "./components/ReportExport.jsx";
import EvaluationPanel from "./components/EvaluationPanel.jsx";
import {
  calibrate,
  createUser,
  getEvaluation,
  getMeta,
  getProfile,
  listUsers,
  nextSignal,
  predict,
  resetProfile,
  sendFeedback,
} from "./api.js";

// Every browser gets its own profile, so one visitor's calibration never leaks into another's.
function browserProfileId() {
  const fresh = `guest-${Math.random().toString(36).slice(2, 8)}`;
  try {
    const saved = window.localStorage.getItem("neurogrip-profile");
    if (saved) return saved;
    window.localStorage.setItem("neurogrip-profile", fresh);
  } catch {
    // storage unavailable (private mode, embedded preview): a per-visit id is fine
  }
  return fresh;
}

const DEFAULT_USER = browserProfileId();
const FAULT_LABELS = { dropout: "Electrode dropout", spike: "Amplitude spike", hum: "50 Hz mains hum" };

export default function App() {
  const [meta, setMeta] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [userId, setUserId] = useState(DEFAULT_USER);
  const [users, setUsers] = useState([]);
  const [known, setKnown] = useState([DEFAULT_USER]); // profiles this browser has opened
  const [recording, setRecording] = useState(null);
  const [fault, setFault] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [profile, setProfile] = useState(null);
  const [status, setStatus] = useState("loading"); // idle | loading | error
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");

  const grasps = meta?.grasps || [];
  const graspLabels = useMemo(
    () => Object.fromEntries(grasps.map((g) => [g.key, g.label])),
    [grasps]
  );
  const sourceLabels = useMemo(
    () => Object.fromEntries((meta?.sources || []).map((s) => [s.key, s.label])),
    [meta]
  );

  const fail = (err) => {
    setStatus("error");
    setErrorMsg(err.message || "Something went wrong talking to the API.");
  };

  const runCycle = useCallback(async (forUser, withFault) => {
    setStatus("loading");
    setErrorMsg("");
    setNotice("");
    try {
      const rec = await nextSignal(forUser, withFault);
      setRecording(rec);
      setPrediction(await predict(forUser, rec));
      setStatus("idle");
    } catch (err) {
      fail(err);
    }
  }, []);

  const refresh = useCallback(async (forUser) => {
    try {
      setProfile(await getProfile(forUser)); // creates the profile on first use
      setKnown((ids) => (ids.includes(forUser) ? ids : [...ids, forUser]));
      setUsers((await listUsers()).users);
    } catch (err) {
      fail(err);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [m, e] = await Promise.all([getMeta(), getEvaluation()]);
        setMeta(m);
        setEvaluation(e);
        await refresh(DEFAULT_USER);
        await runCycle(DEFAULT_USER, null);
      } catch (err) {
        fail(err);
      }
    })();
  }, [refresh, runCycle]);

  const handleSwitchUser = async (newUserId, source) => {
    try {
      if (source) await createUser(newUserId, source);
      setUserId(newUserId);
      await refresh(newUserId);
      await runCycle(newUserId, fault);
    } catch (err) {
      fail(err);
    }
  };

  const handleFeedback = async (actualGrasp) => {
    if (!recording) return;
    try {
      const updated = await sendFeedback(userId, recording, actualGrasp);
      setProfile(updated);
      setNotice(updated.learned ? "Profile updated from this recording." : updated.note);
      if (updated.learned) await runCycle(userId, fault);
    } catch (err) {
      fail(err);
    }
  };

  const handleCalibrate = async () => {
    setStatus("loading");
    try {
      setProfile(await calibrate(userId, 1));
      await runCycle(userId, fault);
      setNotice("Calibrated on one more recording of each grasp.");
    } catch (err) {
      fail(err);
    }
  };

  const handleReset = async () => {
    try {
      setProfile(await resetProfile(userId));
      await runCycle(userId, fault);
      setNotice("Profile reset to the uncalibrated population model.");
    } catch (err) {
      fail(err);
    }
  };

  const busy = status === "loading";

  return (
    <div className="app-shell">
      <header className="hero">
        <span className="eyebrow">NEUROGRIP AI</span>
        <h1>Teaching bionic hands to understand intent, not just signals.</h1>
        <p className="hero-sub">
          Grip-intent classification from real two-channel muscle (EMG) recordings. The model
          explains every decision exactly, adapts to each wearer from a few repetitions, and
          refuses to move the hand when the signal or its own confidence cannot be trusted.
        </p>
      </header>

      {status === "error" && (
        <div className="error-banner" role="alert">
          Couldn&rsquo;t reach the API. If it is hosted on a free tier it may be waking up; wait a
          minute and reload. ({errorMsg})
        </div>
      )}

      <UserSwitcher
        users={users.filter((u) => known.includes(u.user_id))}
        activeUser={userId}
        sources={meta?.sources || []}
        sourceLabels={sourceLabels}
        onSwitch={handleSwitchUser}
      />

      {prediction && <AnomalyBanner anomalies={prediction.anomalies} />}

      <main className="dashboard-grid">
        {recording && <SignalChart recording={recording} />}

        {prediction && <GripDisplay prediction={prediction} graspLabels={graspLabels} />}

        {prediction && <ReasoningPanel prediction={prediction} graspLabels={graspLabels} />}

        <CalibrationChart profile={profile} evaluation={evaluation} grasps={grasps} />
      </main>

      <section className="control-panel">
        <div className="control-row">
          <span className="control-label mono">
            True grasp of this recording:{" "}
            <strong>{recording ? graspLabels[recording.true_grasp] : "…"}</strong>
            {recording?.injected_fault && ` · injected fault: ${FAULT_LABELS[recording.injected_fault]}`}
          </span>
          <button className="btn btn-primary" onClick={() => runCycle(userId, fault)} disabled={busy}>
            {busy ? "Loading…" : "Next recording"}
          </button>
        </div>

        <div className="control-row">
          <span className="control-label">Calibration: the wearer performs each grasp once more</span>
          <div className="intent-buttons">
            <button className="btn btn-primary" onClick={handleCalibrate} disabled={busy}>
              Calibrate (+1 recording per grasp)
            </button>
            <button className="btn btn-ghost" onClick={handleReset} disabled={busy}>
              Reset profile
            </button>
          </div>
        </div>

        <div className="control-row">
          <span className="control-label">Corrupt the next recording to test the safety gate</span>
          <div className="intent-buttons">
            <button
              className={`btn btn-ghost ${fault === null ? "btn-suggested" : ""}`}
              onClick={() => setFault(null)}
            >
              Clean signal
            </button>
            {(meta?.fault_types || []).map((f) => (
              <button
                key={f}
                className={`btn btn-ghost ${fault === f ? "btn-suggested" : ""}`}
                onClick={() => setFault(f)}
              >
                {FAULT_LABELS[f] || f}
              </button>
            ))}
          </div>
        </div>

        <div className="control-row">
          <span className="control-label">
            Confirm what the grasp really was (the profile learns from it, then the next recording loads)
          </span>
          <div className="intent-buttons">
            {grasps.map((g) => (
              <button
                key={g.key}
                title={g.description}
                className={`btn btn-ghost ${recording?.true_grasp === g.key ? "btn-suggested" : ""}`}
                onClick={() => handleFeedback(g.key)}
                disabled={!recording || busy}
              >
                {g.label}
              </button>
            ))}
          </div>
          {notice && <span className="feedback-ack mono">{notice}</span>}
        </div>

        <div className="control-row">
          <span className="control-label">Export a summary of this wearer&rsquo;s session</span>
          <ReportExport userId={userId} graspLabels={graspLabels} sourceLabels={sourceLabels} />
        </div>
      </section>

      {evaluation && <EvaluationPanel evaluation={evaluation} />}

      <footer className="footer">
        <span>
          Recordings are replayed from the public &ldquo;sEMG for Basic Hand movements&rdquo; dataset
          (Sapsanis, Tzes &amp; Georgoulas, UCI, CC BY 4.0). Research prototype, not a medical device
          and not connected to hardware.
        </span>
      </footer>
    </div>
  );
}
