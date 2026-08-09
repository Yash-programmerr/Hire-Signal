# HireSignal Candidate Ranking

The application uses FastAPI for the backend and React/Vite for the frontend.

## 1.Run the backend

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

## 2.Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and communicates with the FastAPI service at `http://localhost:8000/api`.

## 3.Add files manually or from the Dataset Library section and rank Candidates against a specific JD.

## 4.Happy Ranking :)

## 5.API workflow

1. `POST /api/rankings` uploads the candidates JSONL, job description, and optional YAML metadata.
2. `GET /api/rankings/{job_id}/status` reports live background-job progress.
3. `GET /api/rankings/{job_id}/results` returns server-ranked, sortable and paginated results.
4. `GET /api/rankings/{job_id}/candidates/{candidate_id}` returns a backend-supplied detail profile.
5. `GET /api/rankings/{job_id}/downloads/{kind}` downloads submission, audit, or all output files.
