from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SUBMISSION_CSV   = "./Sheild_Clan.csv"
SCORED_FEATURES  = "./data/candidates_scored_features.parquet"
SSL_SCORES       = "./data/phase4c_ssl_scores.parquet"
CANDIDATES_JSONL = "./data/candidates.jsonl"
OUTPUT_CSV       = "./Sheild_Clan.csv"

# JD ke relevant AI/NLP skills — tier D + tier C from feature_engineering
AI_CORE_SKILLS = {
    # Tier D — direct JD match
    "rag", "faiss", "vector database", "llm", "large language model",
    "langchain", "llama", "gpt", "bert", "transformers", "hugging face",
    "huggingface", "nlp", "natural language processing", "semantic search",
    "elasticsearch", "solr", "lucene", "bm25", "dense retrieval",
    "information retrieval", "search ranking", "learning to rank",
    "sentence transformers", "embeddings", "fine-tuning", "finetuning",
    "rlhf", "prompt engineering", "vector search", "ann", "hnsw",
    # Tier C — adjacent
    "pytorch", "tensorflow", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "deep learning", "neural network", "mlops", "mlflow",
    "kubeflow", "airflow", "spark", "pyspark", "recommendation system",
    "collaborative filtering", "a/b testing", "python", "machine learning",
}

NLP_IR_SKILLS = {
    "Hugging Face Transformers","LangChain","Information Retrieval","LLMs","Recommendation Systems",
    "Semantic Search","Sentence Transformers","Embeddings","Vector Search","Prompt Engineering",
    "Pinecone","FAISS","RAG","Fine-tuning LLMs",
    "QLoRA","pgvector","Weaviate","Milvus","Learning to Rank","BM25","TensorFlow","Qdrant","Python",
    "PyTorch","PEFT","LoRA","NLP","Machine Learning","Deep Learning","Haystack","Elasticsearch",
    "LlamaIndex","scikit-learn","OpenSearch",
    "Information Retrieval Systems","Search Backend","Text Encoders","Vector Representations",
    "Content Matching","Model Adaptation","Ranking Systems","Search & Discovery","Workflow Orchestration",
    "Search Infrastructure","Indexing Algorithms","Open-source ML libraries",
    "Natural Language Processing","Document Processing"
}

def pick_best_assessment(scores: dict):
    """Highlight the JD-relevant assessment first, NOT just the max raw number."""
    if not scores:
        return None, None, False
    relevant = {k: v for k, v in scores.items() if k in NLP_IR_SKILLS}
    if relevant:
        skill = max(relevant, key=relevant.get)
        return skill, relevant[skill], True
    skill = max(scores, key=scores.get)   # fallback only if nothing relevant exists
    return skill, scores[skill], False

def build_caveats(notice_days, response_hours, response_rate):
    """Honest-concerns clause -- Stage 4 explicitly checks for this."""
    caveats = []
    if notice_days and notice_days > 60:
        caveats.append(f"{notice_days}-day notice period")
    if response_hours and response_hours > 96:
        caveats.append(f"slow recruiter response (~{response_hours/24:.0f}d)")
    if response_rate is not None and response_rate < 0.30:
        caveats.append(f"low response rate ({response_rate:.2f})")
    return caveats

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Submission loaded: {len(df):,} rows")
    return df


def load_features(scored_path: str, ssl_path: str) -> pd.DataFrame:
    scored = pd.read_parquet(scored_path)
    ssl    = pd.read_parquet(ssl_path)

    # Merge on candidate_id
    keep_ssl = ["candidate_id", "ssl_final_score", "ssl_pred_text",
                "ssl_pred_tabular", "any_hard_disqualifier", "confirmed_honeypot"]
    keep_ssl = [c for c in keep_ssl if c in ssl.columns]

    features = scored.merge(ssl[keep_ssl], on="candidate_id", how="left")
    print(f"✅ Features loaded: {len(features):,} candidates")
    return features


def load_raw_candidates(jsonl_path: str) -> dict:
    """
    Load candidates.jsonl into a dict: {candidate_id: candidate_dict}
    Only load fields needed for reasoning to save memory.
    """
    needed_fields = {
        "candidate_id", "redrob_signals", "skills",
        "profile", "education", "career_history"
    }
    candidates = {}
    path = Path(jsonl_path)
    if not path.exists():
        print(f"⚠️  {jsonl_path} not found — will use parquet data only")
        return {}

    print(f"⏳ Loading candidates.jsonl (this may take ~30s)...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line.strip())
                cid = c.get("candidate_id", "")
                if cid:
                    candidates[cid] = c
            except json.JSONDecodeError:
                continue

    print(f"✅ Raw candidates loaded: {len(candidates):,}")
    return candidates


# ──────────────────────────────────────────────
# HELPER EXTRACTORS
# ──────────────────────────────────────────────
def count_ai_skills(candidate: dict) -> int:
    """Count how many AI core skills the candidate has."""
    skills = candidate.get("skills", [])
    count  = 0
    for skill in skills:
        name = str(skill.get("name", "")).lower()
        if any(ai_skill in name for ai_skill in AI_CORE_SKILLS):
            count += 1
    return count


def get_response_signal(signals: dict) -> str:
    """Convert recruiter_response_rate to human-readable string."""
    rate = signals.get("recruiter_response_rate", None)
    avg_time = signals.get("avg_response_time_hours", None)

    if rate is None:
        return ""

    if avg_time is not None:
        if avg_time < 1:
            return "responds in <1hr"
        elif avg_time < 6:
            return f"responds in ~{int(avg_time)}hr"
        elif avg_time < 24:
            return "responds same day"
        elif avg_time < 48:
            return "responds next day"
        else:
            return f"responds in {int(avg_time//24)}d"

    # Fallback to rate
    if rate >= 0.90:
        return "highly responsive"
    elif rate >= 0.70:
        return "responsive"
    elif rate >= 0.40:
        return "moderately responsive"
    else:
        return "low responsiveness"


def get_location_signal(signals: dict, profile: dict) -> str:
    """Get location string."""
    loc = profile.get("location", "") or signals.get("current_location", "")
    if not loc:
        return ""
    # Shorten to city only
    city = str(loc).split(",")[0].strip()
    relocate = signals.get("willing_to_relocate", False)
    if city in ("Pune", "Noida"):
        return city  # Already in preferred location
    elif relocate:
        return f"{city} (open to relocate)"
    return city


def get_notice_period(signals: dict) -> str:
    days = signals.get("notice_period_days", None)
    if days is None:
        return ""
    if days == 0:
        return "immediate joiner"
    elif days <= 15:
        return f"notice: {int(days)}d"
    elif days <= 30:
        return "notice: 30d"
    elif days <= 60:
        return "notice: 60d"
    elif days <= 90:
        return "notice: 90d"
    else:
        return f"notice: {int(days)}d"


def get_github_signal(signals: dict) -> str:
    gh = signals.get("github_activity_score", -1)
    if gh < 0:
        return ""
    elif gh >= 80:
        return "high GitHub activity"
    elif gh >= 50:
        return "active GitHub"
    return ""


def get_education_signal(candidate: dict) -> str:
    edu_list = candidate.get("education", [])
    if not edu_list:
        return ""
    # Take most recent / highest education
    for edu in edu_list:
        tier   = edu.get("tier", "")
        degree = edu.get("degree", "")
        field  = edu.get("field_of_study", "")
        if tier == "tier_1":
            return f"Tier-1 institution ({degree})" if degree else "Tier-1 institution"
        elif tier == "tier_2":
            return f"Tier-2 institution" if not degree else ""
    return ""


def get_open_to_work(signals: dict) -> str:
    otw = signals.get("open_to_work", False)
    return "actively looking" if otw else ""


def get_assessment_signal(signals: dict) -> str:
    """Get top skill assessment score if impressive."""
    scores = signals.get("skill_assessment_scores", {})
    if not scores:
        return ""
    best_score = max(scores.values()) if scores else 0
    best_skill = max(scores, key=scores.get) if scores else ""

    if best_score >= 85:
        return f"top assessments ({best_skill}: {int(best_score)})"
    elif best_score >= 70:
        return f"good assessments ({best_skill}: {int(best_score)})"
    return ""


# ──────────────────────────────────────────────
# MAIN REASONING GENERATOR
# ──────────────────────────────────────────────
def generate_personalized_reasoning(
    row: pd.Series,
    features_row: pd.Series | None,
    candidate: dict | None,
) -> str:
    """
    Generate a candidate-specific reasoning string.

    Target format (matching sample_submission):
      "ML Engineer with 7.2 yrs; 6 AI core skills; responds in <1hr; Pune; notice: 15d"

    For disqualified:
      "Disqualified: [specific reason]"
    """

    # ── Disqualified ──────────────────────────
    if features_row is not None:
        if features_row.get("confirmed_honeypot", False):
            return "Disqualified: honeypot candidate detected"
        if features_row.get("disq_cv_speech_only", False):
            return "Disqualified: no NLP/AI experience (CV/Speech domain only)"
        if features_row.get("disq_consulting_only", False):
            return "Disqualified: consulting-only background, no product ML experience"
        if features_row.get("any_hard_disqualifier", False):
            return "Disqualified: does not meet minimum JD requirements"

    # ── Base info from features parquet ───────
    title = ""
    yoe   = None

    if features_row is not None:
        title = str(features_row.get("current_title", "") or "")
        yoe   = features_row.get("years_of_experience", None)

    # Override with raw data if available
    signals = {}
    profile = {}
    if candidate:
        signals  = candidate.get("redrob_signals", {}) or {}
        profile  = candidate.get("profile", {}) or {}
        if not title:
            title = profile.get("current_title", "") or profile.get("headline", "") or ""
        if yoe is None:
            yoe = profile.get("years_of_experience", None)

    # ── Build parts ───────────────────────────
    parts = []

    # 1. Title + YOE (always first — matches sample format)
    if title and yoe is not None:
        parts.append(f"{title} with {float(yoe):.1f} yrs")
    elif title:
        parts.append(title)
    elif yoe is not None:
        parts.append(f"{float(yoe):.1f} yrs experience")

    # 2. AI skill count
    if candidate:
        ai_count = count_ai_skills(candidate)
        if ai_count > 0:
            parts.append(f"{ai_count} AI core skills")
    elif features_row is not None:
        # Fallback: use tier scores as proxy
        tier_d = features_row.get("num_tier_d_skills", None)
        tier_c = features_row.get("num_tier_c_skills", None)
        if tier_d is not None:
            total = int(tier_d) + int(tier_c or 0)
            if total > 0:
                parts.append(f"{total} AI core skills")

    # 3. Responsiveness (from signals)
    if signals:
        resp = get_response_signal(signals)
        if resp:
            parts.append(resp)

    # 4. Location
    if signals or profile:
        loc = get_location_signal(signals, profile)
        if loc:
            parts.append(loc)

    # 5. Notice period
    if signals:
        notice = get_notice_period(signals)
        if notice:
            parts.append(notice)

    # 6. Extra differentiators (add if space / quality indicator)
    if signals:
        gh = get_github_signal(signals)
        if gh:
            parts.append(gh)

        otw = get_open_to_work(signals)
        if otw:
            parts.append(otw)

    if candidate:
        edu = get_education_signal(candidate)
        if edu:
            parts.append(edu)

        assess = get_assessment_signal(signals)
        if assess:
            parts.append(assess)

    # ── Fallback if nothing built ──────────────
    if not parts:
        score = row.get("score", 0)
        if score >= 0.85:
            return "Strong overall fit for NLP/Search Ranking role"
        elif score >= 0.60:
            return "Good fit for NLP/Search Ranking role"
        else:
            return "Below-average fit for NLP/Search Ranking role"

    return "; ".join(parts)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Personalized Reasoning Generator")
    print("=" * 60)

    # Load all data
    submission = load_submission(SUBMISSION_CSV)
    features   = load_features(SCORED_FEATURES, SSL_SCORES)
    raw        = load_raw_candidates(CANDIDATES_JSONL)

    # Index features by candidate_id for fast lookup
    feat_index = features.set_index("candidate_id")

    print(f"\n⏳ Generating personalized reasoning for {len(submission):,} candidates...")

    new_reasonings = []
    for i, row in submission.iterrows():
        cid = row["candidate_id"]

        # Get features row
        feat_row = feat_index.loc[cid] if cid in feat_index.index else None

        # Get raw candidate dict
        cand = raw.get(cid, None)

        reasoning = generate_personalized_reasoning(row, feat_row, cand)
        new_reasonings.append(reasoning)

        if (i + 1) % 10_000 == 0:
            print(f"   Processed {i+1:,} / {len(submission):,}...")

    submission["reasoning"] = new_reasonings

    # Stats
    unique_count = submission["reasoning"].nunique()
    print(f"\n✅ Reasoning generated:")
    print(f"   Unique strings: {unique_count:,} (was 185 before)")
    print(f"   Coverage      : {unique_count/len(submission):.1%} uniqueness")

    # Preview top 20
    print(f"\n📋 Sample reasoning — Top 20:")
    print(f"   {'Rank':>4}  {'candidate_id':<15}  reasoning")
    print(f"   {'─'*75}")
    for _, row in submission.head(20).iterrows():
        print(f"   {int(row['rank']):>4}  {row['candidate_id']:<15}  {row['reasoning'][:65]}")

    # Preview disqualified
    disq_sample = submission[submission["reasoning"].str.startswith("Disqualified")].head(5)
    print(f"\n   Disqualified sample:")
    for _, row in disq_sample.iterrows():
        print(f"   Rank {int(row['rank']):>6}: {row['reasoning']}")

    # Save
    submission.to_csv(OUTPUT_CSV, index=False)
    size_kb = Path(OUTPUT_CSV).stat().st_size / 1024
    print(f"\n💾 Saved → {OUTPUT_CSV}  ({len(submission):,} rows, {size_kb:.1f} KB)")
    print(f"\n✅ Done! Submit {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
