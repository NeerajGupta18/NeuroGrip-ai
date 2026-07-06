const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8123";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

export function generateSignal(intent = null, injectAnomaly = null) {
  return request("/api/signal/generate", {
    method: "POST",
    body: JSON.stringify({ intent, inject_anomaly: injectAnomaly }),
  });
}

export function predict(signal, userId) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify({ signal, user_id: userId }),
  });
}

export function sendFeedback({ userId, signal, predictedIntent, actualIntent }) {
  return request("/api/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      signal,
      predicted_intent: predictedIntent,
      actual_intent: actualIntent,
    }),
  });
}

export function getProfile(userId) {
  return request(`/api/profile/${userId}`, { method: "GET" });
}

export function listUsers() {
  return request("/api/users", { method: "GET" });
}

export function getReport(userId) {
  return request(`/api/report/${userId}`, { method: "GET" });
}
