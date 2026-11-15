# ---- 1. build the dashboard ----
FROM node:20-alpine AS dashboard
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# No VITE_API_BASE: the dashboard calls the API on its own origin.
RUN npm run build

# ---- 2. the API, which also serves the dashboard ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/app ./app
COPY backend/data ./data
COPY --from=dashboard /frontend/dist /srv/frontend/dist

RUN useradd --create-home appuser && mkdir -p /srv/profiles && chown appuser /srv/profiles
USER appuser
ENV FRONTEND_DIST=/srv/frontend/dist PROFILES_DIR=/srv/profiles ALLOWED_ORIGINS="" PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
