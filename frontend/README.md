# HireSignal Frontend

Premium React 19/Vite frontend for the candidate-ranking workflow. It contains no ranking, scoring, parsing, or ML logic.

## Run locally

```bash
npm install
npm run dev
```

## Backend integration

Start the FastAPI server from the repository root before running the frontend:

```bash
uvicorn main:app --reload --port 8000
```

The frontend connects to the FastAPI `/api` endpoints for upload, status polling, results, candidate details, and downloads. Copy `.env.example` to `.env` only when the API is hosted at a different URL.
