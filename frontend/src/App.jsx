import { useCallback, useEffect, useState } from "react";
import SignalChart from "./components/SignalChart.jsx";
import GripDisplay from "./components/GripDisplay.jsx";
import ReasoningPanel from "./components/ReasoningPanel.jsx";
import CalibrationChart from "./components/CalibrationChart.jsx";
import AnomalyBanner from "./components/AnomalyBanner.jsx";
import UserSwitcher from "./components/UserSwitcher.jsx";
import ReportExport from "./components/ReportExport.jsx";
import { generateSignal, predict, sendFeedback, getProfile, listUsers } from "./api.js";

const INTENT_OPTIONS = [
  { key: "grasp", label: "Grasp" },
  { key: "release", label: "Release" },
  { key: "hold_firm", label: "Hold-Firm" },
  { key: "gentle_hold", label: "Hold-Gentle" },
  { key: "pinch", label: "Pinch" },
];

const ANOMALY_OPTIONS = [
  { key: null, label: "Clean signal" },
  { key: "dropout", label: "Simulate dropout" },
  { key: "spike", label: "Simulate spike" },
  { key: "noisy", label: "Simulate noisy sensor" },
];

const DEFAULT_USER = "demo_user";

export default function App() {
  const [userId, setUserId] = useState(DEFAULT_USER);
  const [users, setUsers] = useState([DEFAULT_USER]);
  const [signal, setSignal] = useState(null);
  const [trueIntent, setTrueIntent] = useState(null);
  const [injectAnomaly, setInjectAnomaly] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [profile, setProfile] = useState({
    session_count: 0,
    accuracy_history: [],
    intent_distribution: {},
  });
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMsg, setErrorMsg] = useState("");
  const [feedbackGiven, setFeedbackGiven] = useState(false);

  const runCycle = useCallback(async (forUserId, anomaly = null) => {
    setStatus("loading");
    setErrorMsg("");
    setFeedbackGiven(false);
    try {
      const gen = await generateSignal(null, anomaly);
      setSignal(gen.signal);
      setTrueIntent(gen.true_intent);
      const pred = await predict(gen.signal, forUserId);
      setPrediction(pred);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setErrorMsg(err.message || "Something went wrong talking to the backend.");
    }
  }, []);

  const refreshProfile = useCallback(async (forUserId) => {
    try {
      const p = await getProfile(forUserId);
      setProfile(p);
    } catch {
      // Non-fatal — calibration chart just stays at its last known state.
    }
  }, []);

  const refreshUsers = useCallback(async () => {
    try {
      const res = await listUsers();
      setUsers(res.users.length > 0 ? res.users : [DEFAULT_USER]);
    } catch {
      // Non-fatal — switcher just shows the default user.
    }
  }, []);

  useEffect(() => {
    runCycle(userId, injectAnomaly);
    refreshProfile(userId);
    refreshUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleGenerateNew = () => {
    runCycle(userId, injectAnomaly);
  };

  const handleSwitchUser = async (newUserId) => {
    setUserId(newUserId);
    await runCycle(newUserId, injectAnomaly);
    await refreshProfile(newUserId);
    refreshUsers();
  };

  const handleFeedback = async (actualIntent) => {
    if (!signal || !prediction) return;
    try {
      const updated = await sendFeedback({
        userId,
        signal,
        predictedIntent: prediction.predicted_intent,
        actualIntent,
      });
      setProfile(updated);
      setFeedbackGiven(true);
      refreshUsers();
    } catch (err) {
      setErrorMsg(err.message || "Could not save feedback.");
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <span className="eyebrow">NEUROGRIP AI</span>
        <h1>Teaching bionic hands to understand intent, not just signals.</h1>
        <p className="hero-sub">
          An explainable AI pipeline that reads simulated myoelectric (muscle) signals,
          predicts bionic-hand grip intent, adapts to each user over repeated sessions, and
          flags its own sensor faults — a compact prototype of the reasoning layer behind
          human-augmentation systems like Kawatek&rsquo;s RYO / INTELIHAND.
        </p>
      </header>

      {status === "error" && (
        <div className="error-banner">
          Couldn&rsquo;t reach the backend at the configured API URL. Make sure the FastAPI
          server is running (see README) — {errorMsg}
        </div>
      )}

      <UserSwitcher users={users} activeUser={userId} onSwitch={handleSwitchUser} />

      {prediction && <AnomalyBanner anomalies={prediction.anomalies} />}

      <main className="dashboard-grid">
        {signal && <SignalChart signal={signal} />}

        {prediction && (
          <GripDisplay
            intent={prediction.predicted_intent}
            confidence={prediction.confidence}
          />
        )}

        {prediction && (
          <ReasoningPanel
            reasoning={prediction.reasoning}
            confidenceByIntent={prediction.confidence_by_intent}
            features={prediction.features}
          />
        )}

        <CalibrationChart
          accuracyHistory={profile.accuracy_history || []}
          sessionCount={profile.session_count || 0}
        />
      </main>

      <section className="control-panel">
        <div className="control-row">
          <span className="control-label mono">
            Ground truth (simulated): <strong>{trueIntent}</strong>
          </span>
          <button
            className="btn btn-primary"
            onClick={handleGenerateNew}
            disabled={status === "loading"}
          >
            {status === "loading" ? "Generating…" : "Generate new signal"}
          </button>
        </div>

        <div className="control-row">
          <span className="control-label">Signal condition (for demoing the safety layer)</span>
          <div className="intent-buttons">
            {ANOMALY_OPTIONS.map((opt) => (
              <button
                key={opt.label}
                className={`btn btn-ghost ${injectAnomaly === opt.key ? "btn-suggested" : ""}`}
                onClick={() => setInjectAnomaly(opt.key)}
                disabled={status === "loading"}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-row">
          <span className="control-label">Was this the correct grip intent?</span>
          <div className="intent-buttons">
            {INTENT_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                className={`btn btn-ghost ${prediction?.predicted_intent === opt.key ? "btn-suggested" : ""}`}
                onClick={() => handleFeedback(opt.key)}
                disabled={!prediction || status === "loading"}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {feedbackGiven && <span className="feedback-ack mono">✓ calibration updated</span>}
        </div>

        <div className="control-row">
          <span className="control-label">Export a summary of this user&rsquo;s session history</span>
          <ReportExport userId={userId} />
        </div>
      </section>

      <footer className="footer">
        <span>
          Simulated signals — not a hardware-connected medical device. Built to demonstrate the
          AI reasoning, safety, calibration, and reporting layers end to end.
        </span>
      </footer>
    </div>
  );
}
