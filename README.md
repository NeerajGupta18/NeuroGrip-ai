<div align="center">

# 🧠 NeuroGrip AI

### Teaching bionic hands to understand *intent*, not just signals.

An explainable AI pipeline that reads simulated myoelectric (muscle) signals, predicts bionic-hand grip intent in real time, flags its own sensor faults, and adapts to every individual user — a compact, working prototype of the reasoning layer behind human-augmentation systems like Kawatek's **RYO** / **INTELIHAND**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-1C3C3C?style=flat-square)](https://www.langchain.com/langgraph)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](#license)

</div>

---

## 📋 Table of contents

- [The problem](#-the-problem)
- [What it does](#-what-it-does)
- [Honest scope](#-honest-scope)
- [Architecture](#-architecture)
- [Tech stack](#-tech-stack)
- [Running it locally](#-running-it-locally)
- [API reference](#-api-reference)
- [Design notes](#-design-notes)
- [Roadmap](#-roadmap)

---

## 🎯 The problem

Bionic hands read electrical signals from a wearer's muscles and have to decide, in real time, what the person intended to do — grasp, release, hold gently, hold firm, pinch. Two things make this genuinely hard:

| Challenge | Why it's hard |
|---|---|
| **Noisy signals** | A raw muscle signal is a messy, fast-changing waveform — not an obvious "grab" or "let go" |
| **Every user is different** | A system that isn't personalized to the individual wearer will misfire constantly |
| **Trust** | A prediction with no explanation is a black box — clinicians and users need to know *why*, not just *what* |
| **Sensor faults happen** | Loose electrodes, motion artifacts, and interference are routine in real hardware — a safe system has to know when *not* to trust itself |

Real systems like Kawatek's **INTELIHAND** solve this with AI that learns each user's signal patterns over time. This project builds a working, simplified version of that exact pipeline — signal → interpretation → explanation → personalization → safety.

---

## ✨ What it does

| # | Feature | What it proves |
|---|---|---|
| 1 | **Multi-grip signal simulation** | Generates realistic myoelectric windows for 5 grip intents (grasp, release, hold-firm, hold-gentle, pinch), modeled on real EMG envelope/frequency behaviour |
| 2 | **Explainable LangGraph agent** | `extract_features → detect_anomaly → classify_intent → explain_prediction` — every prediction ships with a plain-English reason, not just a confidence score |
| 3 | **Adaptive per-user calibration** | Confirming/correcting a prediction nudges that user's profile; a live chart shows accuracy climbing over sessions — literally "the AI learns your rhythm" |
| 4 | **Safety anomaly detection** | Flags signal dropout, spikes, and excessive noise *independently* of grip classification, and marks affected predictions as provisional — mirrors the health-monitoring instinct behind real products like RYO's glucose alerts |
| 5 | **Session report export** | One-click downloadable `.txt` summary of accuracy trend + grip distribution — built for the same audience (a clinician reviewing remote rehab progress) Kawatek's own pitch targets |
| 6 | **Multi-user profiles** | Switch between users live and watch calibration genuinely re-personalize — proof the system generalizes, not a hardcoded demo |
| 7 | **Live React dashboard** | Real-time signal chart, grip visualization, confidence breakdown, feature readout, anomaly banner, and calibration curve — all wired to a real backend, not mocked |

<div align="center">
<sub>💡 Add your own dashboard screenshots or a short screen recording here — this is the single highest-impact thing you can do to this README.</sub>
</div>

---

## 🔍 Honest scope

This is a **simulated-signal prototype of the AI reasoning layer** — it is not connected to real EMG hardware and is not a medical device. That's a deliberate choice: it isolates and demonstrates the hardest and most transferable part of the problem (turning noisy signal features into a confident, explainable, personalized, *safety-aware* decision) without needing lab equipment. The signal generation, feature extraction, anomaly detection, and calibration approach are all built the way a real system would be, so the pipeline is a genuine, extensible starting point for hardware integration.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              React Dashboard             │
                    │  Signal chart · Grip display · Reasoning │
                    │  Anomaly banner · Calibration · Reports  │
                    └────────────────────┬──────────────────────┘
                                         │  REST (JSON)
                    ┌────────────────────▼──────────────────────┐
                    │                FastAPI Service             │
                    │                                            │
                    │   /api/signal/generate   /api/predict      │
                    │   /api/feedback          /api/profile      │
                    │   /api/users             /api/report       │
                    └────────────────────┬──────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────┐
                    │           LangGraph Reasoning Agent        │
                    │                                            │
                    │  extract_features → detect_anomaly         │
                    │       → classify_intent → explain          │
                    └────────────────────┬──────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────┐
                    │     Calibration Store (per-user JSON)      │
                    │   centroids · accuracy history · counts    │
                    └────────────────────────────────────────────┘
```

```
neurogrip-ai/
├── backend/                    FastAPI service
│   └── app/
│       ├── signal_simulator.py    synthetic EMG-like signal + fault injection
│       ├── features.py            MAV / VAR / RMS / WL / ZC extraction
│       ├── anomaly.py             dropout / spike / noise safety checks
│       ├── agent.py               LangGraph reasoning pipeline
│       ├── calibration.py         per-user adaptive profiles + distributions
│       ├── models.py              Pydantic request/response schemas
│       └── main.py                API routes
└── frontend/                   React + Vite dashboard
    └── src/
        ├── components/
        │   ├── SignalChart.jsx
        │   ├── GripDisplay.jsx
        │   ├── ReasoningPanel.jsx
        │   ├── CalibrationChart.jsx
        │   ├── AnomalyBanner.jsx
        │   ├── UserSwitcher.jsx
        │   └── ReportExport.jsx
        ├── api.js
        └── App.jsx
```

---

## 🛠️ Tech stack

**Backend** — Python · FastAPI · LangGraph · NumPy · Pydantic
**Frontend** — React · Vite · Recharts
**Optional** — Groq API (Llama 3.3 70B) for richer natural-language reasoning; the project runs fully without it via a deterministic template fallback

---

## 🚀 Running it locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8123
```

API docs (interactive): **http://127.0.0.1:8123/docs**

> Optional: set a `GROQ_API_KEY` environment variable to switch the reasoning step to LLM-generated explanations.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open **http://localhost:5173**

---

## 📡 API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/signal/generate` | `POST` | Generate a simulated signal window for a given (or random) intent, optionally with an injected fault (`dropout` / `spike` / `noisy`) |
| `/api/predict` | `POST` | Run the LangGraph agent: check signal quality, classify intent, explain reasoning |
| `/api/feedback` | `POST` | Confirm/correct a prediction, updating the user's calibration profile |
| `/api/profile/{user_id}` | `GET` | Retrieve a user's calibration state, accuracy history, and grip-intent distribution |
| `/api/users` | `GET` | List all user IDs with a saved calibration profile |
| `/api/report/{user_id}` | `GET` | Session report data (accuracy trend + grip distribution) behind the downloadable report |

---

## 🎨 Design notes

The dashboard is styled as a clinical signal-monitor: graphite background, a teal signal trace, amber for confidence/attention — deliberately avoiding the generic "AI demo" look in favor of something that reads like real instrument UI. Every predicted grip renders as an abstract finger-extension bar chart rather than a literal hand illustration, keeping the focus on the actual signal-to-decision pipeline.

---

## 🗺️ Roadmap

- [ ] Real hardware/sensor integration (replacing the simulator with an actual EMG input)
- [ ] Persisted report history (currently generated on demand from the live profile)
- [ ] Authentication for multi-user profiles (currently open, demo-only)

---

<div align="center">
<sub>Built by Neeraj Gupta · Final-year B.Tech, AI & Data Science, MIT-WPU Pune</sub>
</div>
