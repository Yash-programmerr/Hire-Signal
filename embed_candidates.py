from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
NUM_CANDIDATES = 100_000
BATCH_SIZE = 64
JD_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _safe(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def build_candidate_text(record: dict) -> str:
    profile = record.get("profile") or {}
    skills = record.get("skills") or []
    career_history = record.get("career_history") or []

    current_title = _safe(profile.get("current_title"))
    current_company = _safe(profile.get("current_company"))
    title_at_company = (
        f"{current_title} at {current_company}"
        if current_title or current_company
        else ""
    )

    skill_names = " | ".join(_safe(s.get("name")) for s in skills if s.get("name"))

    job_parts = []
    for job in career_history:
        title = _safe(job.get("title"))
        company = _safe(job.get("company"))
        description = _safe(job.get("description"))
        job_parts.append(f"{title} at {company}: {description}")
    jobs_text = " ".join(job_parts)

    return "\n".join(
        [
            _safe(profile.get("headline")),
            _safe(profile.get("summary")),
            title_at_company,
            skill_names,
            jobs_text,
        ]
    )


def truncate_to_max_tokens(text: str, tokenizer, max_length: int = 512) -> str:
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def stream_candidate_texts(jsonl_path: str) -> dict[str, str]:
    import json

    texts: dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8", buffering=1 << 20) as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            candidate_id = record["candidate_id"]
            texts[candidate_id] = build_candidate_text(record)
    return texts


def load_expected_candidate_ids(features_path: str) -> np.ndarray:
    features = pd.read_parquet(features_path, columns=["candidate_id"])
    assert len(features) == NUM_CANDIDATES, (
        f"expected {NUM_CANDIDATES} rows in features parquet, got {len(features)}"
    )
    return features["candidate_id"].astype(str).to_numpy()


def embed_jd(model: SentenceTransformer, jd_path: Path, out_path: Path) -> np.ndarray:
    raw_jd = jd_path.read_text(encoding="utf-8").strip()
    if not raw_jd:
        print(f"ERROR: {jd_path} is empty. Add the job description text before running.")
        sys.exit(1)

    jd_text = JD_QUERY_PREFIX + raw_jd
    jd_embedding = model.encode(
        jd_text,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    np.save(out_path, jd_embedding)
    return jd_embedding


def embed_candidates(
    model: SentenceTransformer,
    jsonl_path: str,
    expected_ids: np.ndarray,
    embeddings_path: Path,
    ids_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    text_by_id = stream_candidate_texts(jsonl_path)

    missing = set(expected_ids) - set(text_by_id)
    extra = set(text_by_id) - set(expected_ids)
    if missing:
        raise ValueError(f"JSONL missing {len(missing)} candidate_ids present in features parquet")
    if extra:
        raise ValueError(f"JSONL has {len(extra)} candidate_ids not in features parquet")

    sorted_ids = np.sort(expected_ids)

    all_embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(sorted_ids), BATCH_SIZE), desc="Embedding batches"):
        batch_ids = sorted_ids[start : start + BATCH_SIZE]
        batch = [
            truncate_to_max_tokens(text_by_id[cid], model.tokenizer) for cid in batch_ids
        ]
        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        all_embeddings.append(batch_emb.astype(np.float32))

    candidate_embeddings = np.vstack(all_embeddings)
    np.save(embeddings_path, candidate_embeddings)
    np.save(ids_path, sorted_ids)
    return candidate_embeddings, sorted_ids


def compute_cosine_similarities(
    jd_embedding: np.ndarray, candidate_embeddings: np.ndarray
) -> np.ndarray:
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    normed = candidate_embeddings / np.clip(norms, 1e-9, None)
    jd_normed = jd_embedding / np.linalg.norm(jd_embedding)
    return (normed @ jd_normed).astype(np.float32)


def save_preview(
    candidate_ids: np.ndarray,
    similarities: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    preview_df = pd.DataFrame(
        {"candidate_id": candidate_ids, "semantic_similarity": similarities}
    )
    preview_df = preview_df.sort_values("semantic_similarity", ascending=False)
    top20 = preview_df.head(20)
    bottom20 = preview_df.tail(20)
    combined = pd.concat([top20, bottom20], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print("\nSemantic similarity preview (top 20 + bottom 20):")
    print(combined.to_string(index=False))
    return combined


def assert_acceptance(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    embedding_candidate_ids: np.ndarray,
    semantic_similarity_scores: np.ndarray,
    expected_ids: np.ndarray,
) -> None:
    assert jd_embedding.shape == (EMBED_DIM,), f"jd shape {jd_embedding.shape}"
    assert candidate_embeddings.shape == (NUM_CANDIDATES, EMBED_DIM), (
        f"candidate embeddings shape {candidate_embeddings.shape}"
    )
    assert embedding_candidate_ids.shape == (NUM_CANDIDATES,), (
        f"id array shape {embedding_candidate_ids.shape}"
    )
    assert semantic_similarity_scores.shape == (NUM_CANDIDATES,), (
        f"similarity shape {semantic_similarity_scores.shape}"
    )
    assert np.isnan(candidate_embeddings).sum() == 0
    assert np.isnan(semantic_similarity_scores).sum() == 0
    assert semantic_similarity_scores.min() >= -1.0 and semantic_similarity_scores.max() <= 1.0

    expected_set = set(expected_ids)
    actual_set = set(embedding_candidate_ids.tolist())
    assert expected_set == actual_set, "embedding_candidate_ids must match features parquet ids"


def print_runtime_report(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    semantic_similarity_scores: np.ndarray,
    embedding_candidate_ids: np.ndarray,
    features_path: str,
    elapsed_sec: float,
) -> None:
    features = pd.read_parquet(features_path, columns=["candidate_id", "current_title"])
    sim_df = pd.DataFrame(
        {
            "candidate_id": embedding_candidate_ids,
            "semantic_similarity": semantic_similarity_scores,
        }
    )
    joined = sim_df.merge(features, on="candidate_id", how="left")

    print(f"\nTotal candidates embedded: {NUM_CANDIDATES}")
    print(f"JD embedding shape: {jd_embedding.shape}")
    print(f"Candidate embeddings shape: {candidate_embeddings.shape}")

    sim = semantic_similarity_scores
    print(
        "Semantic similarity stats: "
        f"min={sim.min():.4f}, max={sim.max():.4f}, mean={sim.mean():.4f}, "
        f"median={np.median(sim):.4f}, p25={np.percentile(sim, 25):.4f}, "
        f"p75={np.percentile(sim, 75):.4f}"
    )

    top5 = joined.nlargest(5, "semantic_similarity")
    bottom5 = joined.nsmallest(5, "semantic_similarity")

    print("\nTop 5 candidates by similarity:")
    for _, row in top5.iterrows():
        print(
            f"  {row['candidate_id']}  {row['semantic_similarity']:.4f}  "
            f"{row['current_title']}"
        )

    print("\nBottom 5 candidates by similarity:")
    for _, row in bottom5.iterrows():
        print(
            f"  {row['candidate_id']}  {row['semantic_similarity']:.4f}  "
            f"{row['current_title']}"
        )

    print(f"\nTotal wall-clock time: {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f} min)")
    print(
        f"Estimated RAM used (candidate_embeddings): "
        f"{sys.getsizeof(candidate_embeddings) / 1e6:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute JD and candidate embeddings.")
    parser.add_argument("--jsonl", default="./data/candidates.jsonl")
    parser.add_argument("--features", default="./data/candidates_scored_features.parquet")
    parser.add_argument("--jd", default="./data/jd.txt")
    parser.add_argument("--out_dir", default="./data/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all candidates even if cached embeddings exist",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jd_path = Path(args.jd)
    if not jd_path.is_file():
        print(
            f"ERROR: {jd_path} not found. Create this file with the job description "
            "text before running embed_candidates.py"
        )
        sys.exit(1)

    embeddings_path = out_dir / "candidate_embeddings.npy"
    ids_path = out_dir / "embedding_candidate_ids.npy"
    jd_embedding_path = out_dir / "jd_embedding.npy"
    similarity_path = out_dir / "semantic_similarity_scores.npy"
    preview_path = out_dir / "semantic_similarity_preview.csv"

    print("Starting offline precompute — this will take 20–60 min on CPU. Go get a coffee.")

    t0 = time.time()
    expected_ids = load_expected_candidate_ids(args.features)

    skip_embed = (
        not args.force
        and embeddings_path.is_file()
        and ids_path.is_file()
        and np.load(embeddings_path, mmap_mode="r").shape == (NUM_CANDIDATES, EMBED_DIM)
    )

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    if jd_embedding_path.is_file() and not args.force:
        jd_embedding = np.load(jd_embedding_path)
    else:
        print("Embedding JD...")
        jd_embedding = embed_jd(model, jd_path, jd_embedding_path)
        print(f"Saved JD embedding to {jd_embedding_path}")

    if skip_embed:
        print(f"Found cached embeddings at {embeddings_path}; skipping Steps 2–3.")
        candidate_embeddings = np.load(embeddings_path)
        embedding_candidate_ids = np.load(ids_path, allow_pickle=True)
    else:
        print("Embedding candidates...")
        candidate_embeddings, embedding_candidate_ids = embed_candidates(
            model,
            args.jsonl,
            expected_ids,
            embeddings_path,
            ids_path,
        )
        print(f"Saved candidate embeddings to {embeddings_path}")
        print(f"Saved candidate id order to {ids_path}")

    print("Computing cosine similarities...")
    semantic_similarity_scores = compute_cosine_similarities(jd_embedding, candidate_embeddings)
    np.save(similarity_path, semantic_similarity_scores)
    save_preview(embedding_candidate_ids, semantic_similarity_scores, preview_path)

    assert_acceptance(
        jd_embedding,
        candidate_embeddings,
        embedding_candidate_ids,
        semantic_similarity_scores,
        expected_ids,
    )

    elapsed = time.time() - t0
    print_runtime_report(
        candidate_embeddings,
        jd_embedding,
        semantic_similarity_scores,
        embedding_candidate_ids,
        args.features,
        elapsed,
    )
    print(f"\nSaved semantic similarities to {similarity_path}")
    print(f"Saved preview CSV to {preview_path}")


if __name__ == "__main__":
    main()
