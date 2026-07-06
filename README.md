# NeuroGrip AI

**Teaching bionic hands to understand intent, not just signals.**

An explainable AI pipeline that interprets simulated myoelectric (muscle) signals to predict prosthetic-hand grip intent in real time — with per-user adaptive calibration and a live reasoning dashboard.

> Built as a focused prototype of the AI decision layer behind human-augmentation systems (e.g. bionic hands like Kawatek's RYO / INTELIHAND): signal → interpretation → explanation → personalization.

## The problem

Bionic hands read electrical signals from a wearer's muscles and have to decide, in real time, what the person intended to do — grasp, release, hold gently, hold firm, pinch. Two things make this hard:

1. **The signals are noisy.** A raw muscle signal is a messy, fast-changing waveform — not an obvious "grab" or "let go."
2. **Every person's signal is different.** A system that isn't personalized to the individual wearer will misfire constantly.

Real systems like Kawatek's INTELIHAND solve this with AI that learns each user's signal patterns over time. This project builds a working, simplified version of that same pipeline.

## What it does

- **Simulates** myoelectric signal windows for 5 grip intents (grasp, release, hold-firm, hold-gentle, pinch), using signal shapes and noise characteristics modeled after real EMG behaviour.
- **Extracts classic EMG time-domain features** (MAV, VAR, RMS, Waveform Length, Zero-Crossing rate) — the same feature family used in real myoelectric control research.
- **Classifies intent with a LangGraph agent** (`extract_features → detect_anomaly → classify_intent → explain_prediction`), which also explains *why* it made that call in plain English — not just a confidence number.
- **Adapts per user**: confirming or correcting a prediction nudges that user's calibration profile, and a live chart shows classification accuracy improving over sessions — the same "AI learns your rhythm" idea real bionic hands rely on.
- **Flags its own sensor faults**: a dedicated safety layer detects signal dropout, spikes, and excessive noise independently of grip classification, and marks any affected prediction as provisional — mirroring the health-monitoring angle of real systems (e.g. RYO's glucose alerts). The dashboard includes controls to simulate each fault type on demand for demoing.
- **Exports a session report**: a one-click downloadable `.txt` summary of a user's accuracy trend and grip-intent distribution — aimed at the same audience (a clinician or caregiver reviewing remote rehabilitation progress) that Kawatek's own "smart rehabilitation" pitch targets.
- **Supports multiple user profiles**: a profile switcher lets you create/switch between users, showing calibration genuinely personalizes per user rather than being hardcoded to one demo case.
- **Live dashboard (React)**: real-time signal chart, predicted grip visualization, confidence breakdown across all 5 intents, feature readout, anomaly banner, and the calibration learning curve.

## Honest scope

This is a **simulated-signal prototype of the AI reasoning layer** — it is not connected to real EMG hardware and is not a medical device. That's a deliberate choice: it isolates and demonstrates the hardest and most transferable part of the problem (turning noisy signal features into a confident, explainable, personalized decision) without needing lab equipment. The signal generation, feature extraction, and calibration approach are all built the way a real system would be, so the pipeline is a genuine starting point for hardware integration.

## Architecture

```
neurogrip-ai/
├── backend/                 FastAPI service
│   └── app/
│       ├── signal_simulator.py   synthetic EMG-like signal + fault injection
│       ├── features.py           MAV / VAR / RMS / WL / ZC extraction
│       ├── anomaly.py            dropout / spike / noise safety checks
│       ├── agent.py              LangGraph reasoning pipeline
│       ├── calibration.py        per-user adaptive profiles + intent distribution
│       ├── models.py             Pydantic request/response schemas
│       └── main.py               API routes
└── frontend/                React + Vite dashboard
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

## Tech stack

**Backend:** Python, FastAPI, LangGraph, NumPy, Pydantic
**Frontend:** React, Vite, Recharts
**Optional:** Groq API (Llama 3.3 70B) for richer natural-language reasoning — the project runs fully without it via a deterministic template fallback.

## Running it locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8123
```

API docs available at `http://127.0.0.1:8123/docs`.

Optional: set `GROQ_API_KEY` as an environment variable to switch the reasoning step to LLM-generated explanations.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # points the frontend at the backend URL
npm run dev
```

Open `http://localhost:5173`.

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/signal/generate` | POST | Generate a simulated signal window for a given (or random) intent, optionally with an injected fault (`dropout`/`spike`/`noisy`) for demoing the safety layer |
| `/api/predict` | POST | Run the LangGraph agent: check signal quality, classify intent, explain reasoning |
| `/api/feedback` | POST | Confirm/correct a prediction, updating the user's calibration profile |
| `/api/profile/{user_id}` | GET | Retrieve a user's calibration state, accuracy history, and grip-intent distribution |
| `/api/users` | GET | List all user IDs with a saved calibration profile |
| `/api/report/{user_id}` | GET | Session report data (accuracy trend + grip distribution) behind the frontend's downloadable report |

## Roadmap

- Real hardware/sensor integration (replacing the simulator with an actual EMG input)
- Persisted report history (currently generated on demand from the live profile)
- Authentication for multi-user profiles (currently open, demo-only)
