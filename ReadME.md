# HireSignal Candidate Ranking

The application now uses FastAPI for the backend and React/Vite for the frontend. Streamlit has been removed.

## Run the backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and communicates with the FastAPI service at `http://localhost:8000/api`.

## API workflow

1. `POST /api/rankings` uploads the candidates JSONL, job description, and optional YAML metadata.
2. `GET /api/rankings/{job_id}/status` reports live background-job progress.
3. `GET /api/rankings/{job_id}/results` returns server-ranked, sortable and paginated results.
4. `GET /api/rankings/{job_id}/candidates/{candidate_id}` returns a backend-supplied detail profile.
5. `GET /api/rankings/{job_id}/downloads/{kind}` downloads submission, audit, or all output files.
