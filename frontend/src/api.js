const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8123";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : null;
    throw new Error(detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body) });
const get = (path) => request(path, { method: "GET" });

export const getMeta = () => get("/api/meta");
export const getEvaluation = () => get("/api/evaluation");
export const listUsers = () => get("/api/users");
export const createUser = (userId, source) => post("/api/users", { user_id: userId, source });
export const getProfile = (userId) => get(`/api/profile/${encodeURIComponent(userId)}`);
export const resetProfile = (userId) => post(`/api/profile/${encodeURIComponent(userId)}/reset`, {});
export const getReport = (userId) => get(`/api/report/${encodeURIComponent(userId)}`);

export const nextSignal = (userId, injectFault = null) =>
  post("/api/signal/next", { user_id: userId, inject_fault: injectFault });

export const predict = (userId, recording) =>
  post("/api/predict", { user_id: userId, ch1: recording.ch1, ch2: recording.ch2 });

export const sendFeedback = (userId, recording, actualGrasp) =>
  post("/api/feedback", {
    user_id: userId,
    ch1: recording.ch1,
    ch2: recording.ch2,
    actual_grasp: actualGrasp,
  });

export const calibrate = (userId, reps = 1) => post("/api/calibrate", { user_id: userId, reps });
