"""FastAPI service for the HireSignal candidate-ranking workflow."""

from __future__ import annotations

import csv
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scoring import (
    NLP_IR_SKILLS,
    score_candidate,
)

API_PREFIX = "/api"
STORAGE_DIR = Path("/tmp/jobs")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_LIBRARY_DIR = Path(os.getenv("DATASET_LIBRARY_DIR", "dataset_library"))
LIBRARY_CANDIDATES_DIR = Path("/tmp/library/candidates")
LIBRARY_JDS_DIR = DATASET_LIBRARY_DIR / "job_descriptions"
LIBRARY_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_JDS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="HireSignal Ranking API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@dataclass
class RankingJob:
    job_id: str
    directory: Path
    state: Literal["queued", "processing", "completed", "failed"] = "queued"
    progress: int = 0
    current_step: str = "Queued"
    error: str | None = None
    created_at: float = field(default_factory=time.perf_counter)
    processing_time_seconds: float | None = None
    total_candidates: int = 0
    filtered_candidates: int = 0
    average_match_score: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)


jobs: dict[str, RankingJob] = {}
jobs_lock = threading.Lock()


class LibraryRankingRequest(BaseModel):
    candidateId: str
    jobDescriptionId: str


def _job_or_404(job_id: str) -> RankingJob:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ranking job not found")
    return job


def _save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(upload.file, output)


def _validate_extension(upload: UploadFile, allowed: set[str], label: str) -> None:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in allowed:
        raise HTTPException(status_code=422, detail=f"{label} must use one of: {', '.join(sorted(allowed))}")


def _library_files(root: Path, allowed_extensions: set[str]) -> list[dict[str, Any]]:
    root = root.resolve()
    items = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            items.append({"id": path.relative_to(root).as_posix(), "name": path.name, "extension": path.suffix.lower(), "sizeBytes": path.stat().st_size})
    return sorted(items, key=lambda item: item["name"].lower())


def _library_file_or_404(root: Path, identifier: str, allowed_extensions: set[str], label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / identifier).resolve()
    if resolved_root not in candidate.parents or not candidate.is_file() or candidate.suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=404, detail=f"{label} not found in the dataset library")
    return candidate


def _education_summary(record: dict[str, Any]) -> str | None:
    education = record.get("education") or []
    if not education:
        return None
    item = education[0]
    return " · ".join(str(value) for value in [item.get("degree"), item.get("field_of_study"), item.get("institution")] if value)


def _reason_codes(matched_skills: list[str], relevance: float, flags: dict[str, bool]) -> list[str]:
    """Return JD-specific evidence and existing caution flags for every candidate."""
    codes = [f"Profile-to-JD relevance: {relevance * 100:.1f}%"]
    if matched_skills:
        preview = ", ".join(matched_skills[:3])
        suffix = "" if len(matched_skills) <= 3 else f" +{len(matched_skills) - 3} more"
        codes.append(f"JD-matched skills: {preview}{suffix}")

    penalties = {
        "honeypot": "Honeypot signal detected",
        "consulting_only": "Consulting-only career history",
        "cv_speech_only": "CV/Speech profile without NLP/IR signal",
        "aspirational": "Aspirational career pivot signal",
    }
    codes.extend(label for key, label in penalties.items() if flags.get(key))
    return codes


def _to_candidate(record: dict[str, Any], score: float, flags: dict[str, bool], relevance: float, jd_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = record.get("profile") or {}
    candidate_id = str(record.get("candidate_id", ""))
    jd_normalized = jd_text.lower()
    skills = [str(skill.get("name")) for skill in record.get("skills", []) if str(skill.get("name", "")).lower() in jd_normalized]
    timeline = [{"title": str(item.get("title", "Role not provided")), "company": item.get("company"), "dateRange": " – ".join(value for value in [item.get("start_date"), item.get("end_date") or "Present"] if value), "description": item.get("description")} for item in record.get("career_history", [])]
    overview = {"id": candidate_id, "name": str(profile.get("anonymized_name") or candidate_id), "score": score, "matchPercentage": round(score * 100, 2), "experience": f"{float(profile.get('years_of_experience', 0)):.1f} years"}
    explanation = f"{relevance * 100:.1f}% profile-to-job-description relevance"
    if skills:
        explanation += f"; matched skills: {', '.join(skills[:5])}"
    if flags.get("any_disq"):
        explanation += "; ranking reduced by profile integrity or eligibility signals"
    return overview, {**overview, "matchedSkills": skills, "education": _education_summary(record), "reasonCodes": _reason_codes(skills, relevance, flags), "rankingExplanation": explanation, "timeline": timeline}


def _candidate_text(record: dict[str, Any]) -> str:
    profile = record.get("profile") or {}
    skills = " ".join(str(item.get("name", "")) for item in record.get("skills", []))
    career = " ".join(" ".join(str(job.get(field, "")) for field in ("title", "company", "industry", "description")) for job in record.get("career_history", []))
    return " ".join(str(profile.get(field, "")) for field in ("current_title", "current_company", "headline", "summary")) + " " + skills + " " + career


def _extract_job_description(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    else:
        raise ValueError("Unsupported job-description file type")
    if len(text.split()) < 3:
        raise ValueError("Job description must contain meaningful text")
    return text


def _jd_relevance(candidates_path: Path, jd_text: str, job: RankingJob) -> dict[str, float]:
    """Score each profile against this job's own text using TF-IDF cosine similarity."""
    job.current_step = "Analyzing job description"
    candidate_ids: list[str] = []
    candidate_texts: list[str] = []
    with candidates_path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            candidate_ids.append(str(record["candidate_id"]))
            candidate_texts.append(_candidate_text(record))
    job.current_step = "Matching candidate profiles"
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30_000, sublinear_tf=True, dtype=np.float32)
    vectors = vectorizer.fit_transform([jd_text, *candidate_texts])
    similarities = cosine_similarity(vectors[1:], vectors[0]).ravel()
    return dict(zip(candidate_ids, similarities.tolist(), strict=True))


def _write_outputs(job: RankingJob) -> None:
    job.candidates.sort(key=lambda candidate: (-candidate["score"], candidate["id"]))
    for index, candidate in enumerate(job.candidates, start=1):
        candidate["rank"] = index
        job.details[candidate["id"]]["rank"] = index

    submission_path = job.directory / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for candidate in job.candidates[:100]:
            detail = job.details[candidate["id"]]
            writer.writerow({"candidate_id": candidate["id"], "rank": candidate["rank"], "score": candidate["score"], "reasoning": detail["rankingExplanation"]})

    audit_path = job.directory / "ranking_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["candidate_id", "rank", "name", "score", "match_percentage", "experience", "reason_codes", "reasoning"])
        writer.writeheader()
        for candidate in job.candidates:
            detail = job.details[candidate["id"]]
            writer.writerow({"candidate_id": candidate["id"], "rank": candidate["rank"], "name": candidate["name"], "score": candidate["score"], "match_percentage": candidate["matchPercentage"], "experience": candidate["experience"], "reason_codes": "; ".join(detail["reasonCodes"]), "reasoning": detail["rankingExplanation"]})

    with zipfile.ZipFile(job.directory / "ranking_outputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, submission_path.name)
        archive.write(audit_path, audit_path.name)


def _process_job(job: RankingJob, candidates_path: Path, job_description_path: Path) -> None:
    try:
        job.state = "processing"
        job.current_step = "Loading candidates"
        with candidates_path.open("r", encoding="utf-8-sig") as source:
            job.total_candidates = sum(1 for line in source if line.strip())
        if job.total_candidates == 0:
            raise ValueError("Candidates dataset does not contain any JSONL records")

        jd_text = _extract_job_description(job_description_path)
        relevance = _jd_relevance(candidates_path, jd_text, job)
        job.current_step = "Calculating scores"
        total_score = 0.0
        with candidates_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                quality_score, _, flags = score_candidate(record)
                jd_score = relevance.get(str(record["candidate_id"]), 0.0)
                score = 0.80 * jd_score + 0.20 * quality_score
                if flags["any_disq"]:
                    score *= 0.05
                overview, detail = _to_candidate(record, round(score, 6), flags, jd_score, jd_text)
                job.candidates.append(overview)
                job.details[overview["id"]] = detail
                job.filtered_candidates += int(flags["any_disq"])
                total_score += score
                job.progress = min(94, max(1, int(line_number / job.total_candidates * 94)))

        job.average_match_score = round(total_score / len(job.candidates) * 100, 2)
        job.current_step = "Generating submission"
        job.progress = 96
        _write_outputs(job)
        job.processing_time_seconds = round(time.perf_counter() - job.created_at, 2)
        job.progress = 100
        job.current_step = "Output files ready"
        job.state = "completed"
    except Exception as exc:
        job.error = str(exc)
        job.current_step = "Ranking failed"
        job.state = "failed"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{API_PREFIX}/dataset-library/candidates")
def library_candidates() -> dict[str, list[dict[str, Any]]]:
    return {"items": _library_files(LIBRARY_CANDIDATES_DIR, {".jsonl"})}


@app.get(f"{API_PREFIX}/dataset-library/job-descriptions")
def library_job_descriptions() -> dict[str, list[dict[str, Any]]]:
    return {"items": _library_files(LIBRARY_JDS_DIR, {".txt", ".pdf", ".docx"})}


@app.post(f"{API_PREFIX}/rankings", status_code=202)
def create_ranking(candidates: UploadFile = File(...), jobDescription: UploadFile = File(...), metadata: UploadFile | None = File(default=None)) -> dict[str, str]:
    _validate_extension(candidates, {".jsonl"}, "Candidates dataset")
    _validate_extension(jobDescription, {".txt", ".pdf", ".docx"}, "Job description")
    if metadata is not None:
        _validate_extension(metadata, {".yaml", ".yml"}, "Metadata")
    job_id = uuid.uuid4().hex
    directory = STORAGE_DIR / job_id
    directory.mkdir(parents=True, exist_ok=False)
    candidates_path = directory / "candidates.jsonl"
    _save_upload(candidates, candidates_path)
    job_description_path = directory / f"job-description{Path(jobDescription.filename or '').suffix.lower()}"
    _save_upload(jobDescription, job_description_path)
    if metadata is not None:
        _save_upload(metadata, directory / "metadata.yaml")
    job = RankingJob(job_id=job_id, directory=directory)
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=_process_job, args=(job, candidates_path, job_description_path), daemon=True).start()
    return {"jobId": job_id}


@app.post(f"{API_PREFIX}/rankings/from-library", status_code=202)
def create_ranking_from_library(request: LibraryRankingRequest) -> dict[str, str]:
    candidates_path = _library_file_or_404(LIBRARY_CANDIDATES_DIR, request.candidateId, {".jsonl"}, "Candidates dataset")
    job_description_path = _library_file_or_404(LIBRARY_JDS_DIR, request.jobDescriptionId, {".txt", ".pdf", ".docx"}, "Job description")
    job_id = uuid.uuid4().hex
    directory = STORAGE_DIR / job_id
    directory.mkdir(parents=True, exist_ok=False)
    job = RankingJob(job_id=job_id, directory=directory)
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=_process_job, args=(job, candidates_path, job_description_path), daemon=True).start()
    return {"jobId": job_id}


@app.get(f"{API_PREFIX}/rankings/{{job_id}}/status")
def ranking_status(job_id: str) -> dict[str, Any]:
    job = _job_or_404(job_id)
    return {"state": job.state, "progress": job.progress, "currentStep": job.current_step, "error": job.error}


@app.get(f"{API_PREFIX}/rankings/{{job_id}}/results")
def ranking_results(job_id: str, page: int = Query(1, ge=1), page_size: int = Query(10, alias="pageSize", ge=1, le=100), search: str | None = None, sort: Literal["rank", "score", "matchPercentage"] = "rank", direction: Literal["asc", "desc"] = "asc") -> dict[str, Any]:
    job = _job_or_404(job_id)
    if job.state != "completed":
        raise HTTPException(status_code=409, detail="Ranking is not complete")
    candidates = job.candidates
    if search:
        normalized = search.lower()
        candidates = [candidate for candidate in candidates if normalized in candidate["name"].lower() or normalized in candidate["id"].lower()]
    candidates = sorted(candidates, key=lambda candidate: candidate[sort], reverse=direction == "desc")
    total = len(candidates)
    start = (page - 1) * page_size
    return {"summary": {"candidatesUploaded": job.total_candidates, "candidatesRanked": len(job.candidates), "filteredCandidates": job.filtered_candidates, "averageMatchScore": job.average_match_score, "processingTimeSeconds": job.processing_time_seconds}, "candidates": candidates[start:start + page_size], "page": {"page": page, "pageSize": page_size, "total": total, "totalPages": max(1, (total + page_size - 1) // page_size)}}


@app.get(f"{API_PREFIX}/rankings/{{job_id}}/candidates/{{candidate_id}}")
def candidate_detail(job_id: str, candidate_id: str) -> dict[str, Any]:
    detail = _job_or_404(job_id).details.get(candidate_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return detail


@app.get(f"{API_PREFIX}/rankings/{{job_id}}/downloads/{{kind}}")
def download(job_id: str, kind: Literal["submission", "audit", "all"]) -> FileResponse:
    job = _job_or_404(job_id)
    filenames = {"submission": ("submission.csv", "text/csv"), "audit": ("ranking_audit.csv", "text/csv"), "all": ("ranking_outputs.zip", "application/zip")}
    filename, media_type = filenames[kind]
    path = job.directory / filename
    if not path.exists():
        raise HTTPException(status_code=409, detail="Output files are not ready")
    return FileResponse(path, filename=filename, media_type=media_type)
