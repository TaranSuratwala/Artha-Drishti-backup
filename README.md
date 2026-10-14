# Artha Drishti

Artha Drishti is an AI-powered stock market intelligence platform that combines screening, forecasting, and backtesting into one workflow.  
It provides a React frontend for analysis and a Flask backend for data pipelines, strategy engines, and model inference.

## Table of Contents

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Key API Endpoints](#key-api-endpoints)
- [Additional Documentation](#additional-documentation)
- [License](#license)

## Overview

The platform is designed to support end-to-end market analysis:

- ingest and maintain stock market data,
- run multi-strategy screening,
- generate model-driven predictions,
- evaluate ideas with historical backtesting,
- monitor results from a unified dashboard.

## Core Capabilities

- **Multi-strategy screening**: Momentum, Piotroski, Swing, Breakout, Value, and Custom logic.
- **AI prediction engine**: LSTM + attention-based forecasting workflows.
- **Backtesting engine**: Strategy performance evaluation over historical windows.
- **Data and scheduling layer**: Automated pipelines with scheduler-driven updates.
- **Interactive frontend**: Dashboard, watchlist, charting, and portfolio views.

## Technology Stack

### Backend
- Python 3.9+
- Flask API
- PostgreSQL + TimescaleDB
- Optional Redis caching

### Frontend
- React 18
- Vite
- Zustand state management
- Plotly-based charting

## Repository Structure

```text
Artha-Drishti-backup/
├── backend/              # Flask API, scheduler, screening, ML, backtesting
│   ├── application.py    # Backend entrypoint
│   └── README.md         # Backend setup and API details
├── frontend/             # React application
│   ├── src/              # Components, pages, hooks, services
│   └── README.md         # Frontend setup and development notes
├── docs/                 # Static assets for site/theme
└── README.md             # Project overview (this file)
```

## Getting Started

### 1) Start the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env        # Linux/macOS
# copy .env.example .env    # Windows
python application.py
```

Backend health check:

```bash
curl http://localhost:5000/api/health
```

### 2) Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app at `http://localhost:5173`.

## Configuration

### Backend Configuration

Copy `backend/.env.example` to `backend/.env` and configure at minimum:

- `DATABASE_URL`
- `APP_HOST`, `APP_PORT`, `APP_ENV`
- `SECRET_KEY`, `JWT_SECRET_KEY`
- `CORS_ORIGINS`
- scheduler flags such as `SCHEDULER_ENABLED` and `SCHEDULER_AUTO_TRAIN`

Optional keys for external news and market data providers are also supported in the same file.

### Frontend Configuration

Copy `frontend/.env.example` to `frontend/.env.local` and set:

- `VITE_API_BASE_URL`
- `VITE_GOOGLE_CLIENT_ID` (if Google auth is enabled)

## Key API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Service health check |
| `/api/stocks` | GET | Latest stock data |
| `/api/history/<ticker>` | GET | Historical ticker data |
| `/api/screen/momentum` | POST | Momentum screening |
| `/api/screen/piotroski` | POST | Piotroski screening |
| `/api/predict/<ticker>` | POST | Model inference for a ticker |
| `/api/train/<ticker>` | POST | Trigger training/update workflow |
| `/api/backtest/<strategy>` | POST | Strategy backtesting |

## Additional Documentation

- Backend: [`backend/README.md`](backend/README.md)
- Frontend: [`frontend/README.md`](frontend/README.md)
- Quick reference: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)

## License

MIT License
