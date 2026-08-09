from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

NLP_IR_SKILLS = {
    "Hugging Face Transformers",
    "LangChain",
    "Information Retrieval",
    "LLMs",
    "Recommendation Systems",
    "Semantic Search",
    "Sentence Transformers",
    "Embeddings",
    "Vector Search",
    "Prompt Engineering",
    "Pinecone",
    "FAISS",
    "RAG",
    "Fine-tuning LLMs",
    "QLoRA",
    "pgvector",
    "Weaviate",
    "Milvus",
    "Learning to Rank",
    "BM25",
    "TensorFlow",
    "Qdrant",
    "Python",
    "PyTorch",
    "PEFT",
    "LoRA",
    "NLP",
    "Machine Learning",
    "Deep Learning",
    "Haystack",
    "Elasticsearch",
    "LlamaIndex",
    "scikit-learn",
    "OpenSearch",
    "Information Retrieval Systems",
    "Search Backend",
    "Text Encoders",
    "Vector Representations",
    "Content Matching",
    "Model Adaptation",
    "Ranking Systems",
    "Search & Discovery",
    "Workflow Orchestration",
    "Search Infrastructure",
    "Indexing Algorithms",
    "Open-source ML libraries",
    "Natural Language Processing",
    "Document Processing",
}

CV_SPEECH_SKILLS = {
    "YOLO",
    "GANs",
    "OpenCV",
    "ASR",
    "Image Classification",
    "Computer Vision",
    "Speech Recognition",
    "CNN",
    "Diffusion Models",
    "TTS",
    "Object Detection",
}

OTHER_AI_SKILLS_TIER_B = {
    "Feature Engineering",
    "Kubeflow",
    "MLOps",
    "BentoML",
    "Data Science",
    "Reinforcement Learning",
    "MLflow",
    "Time Series",
    "Weights & Biases",
    "Forecasting",
    "Statistical Modeling",
}

TIER_D_SKILLS = {
    "Information Retrieval Systems",
    "Search Backend",
    "Text Encoders",
    "Vector Representations",
    "Content Matching",
    "Model Adaptation",
    "Ranking Systems",
    "Search & Discovery",
    "Workflow Orchestration",
    "Search Infrastructure",
    "Indexing Algorithms",
    "Open-source ML libraries",
    "Natural Language Processing",
    "Document Processing",
}

AI_SKILLS_TIER_BC = (NLP_IR_SKILLS - TIER_D_SKILLS) | CV_SPEECH_SKILLS | OTHER_AI_SKILLS_TIER_B

CONSULTING_FIRMS = {
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
    "HCL",
    "Tech Mahindra",
    "Mphasis",
    "Mindtree",
}

AI_TITLES = {
    "ML Engineer",
    "AI Research Engineer",
    "Data Scientist",
    "Senior Software Engineer (ML)",
    "Computer Vision Engineer",
    "Junior ML Engineer",
    "AI Specialist",
    "Recommendation Systems Engineer",
    "Machine Learning Engineer",
    "Applied ML Engineer",
    "Search Engineer",
    "AI Engineer",
    "Senior Data Scientist",
    "NLP Engineer",
    "Senior NLP Engineer",
    "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer",
    "Senior AI Engineer",
    "Senior Applied Scientist",
    "Lead AI Engineer",
}

SWE_ADJACENT_TITLES = {
    "Software Engineer",
    "Full Stack Developer",
    "Cloud Engineer",
    "Java Developer",
    ".NET Developer",
    "DevOps Engineer",
    "Mobile Developer",
    "Frontend Engineer",
    "QA Engineer",
    "Analytics Engineer",
    "Data Engineer",
    "Data Analyst",
    "Backend Engineer",
    "Senior Data Engineer",
    "Senior Software Engineer",
}

AI_STARTUPS_BIGTECH = {
    "Genpact AI",
    "Sarvam AI",
    "Aganitha",
    "Rephrase.ai",
    "Niramai",
    "Glance",
    "Haptik",
    "Wysa",
    "Krutrim",
    "Saarthi.ai",
    "Verloop.io",
    "Mad Street Den",
    "Yellow.ai",
    "Locobuzz",
    "Observe.AI",
    "Meta",
    "Google",
    "Netflix",
    "Amazon",
    "Microsoft",
    "Salesforce",
    "LinkedIn",
    "Apple",
    "Adobe",
    "Uber",
}

PRODUCT_UNICORNS = {
    "Swiggy",
    "CRED",
    "Razorpay",
    "Zomato",
    "Flipkart",
    "Meesho",
    "InMobi",
    "Nykaa",
    "Zoho",
    "Freshworks",
    "Vedantu",
    "Ola",
    "Paytm",
    "BYJU'S",
    "upGrad",
    "PolicyBazaar",
    "Dream11",
    "PharmEasy",
    "PhonePe",
    "Unacademy",
}

TIER1_CITIES = {
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Gurgaon",
    "Pune",
    "Noida",
}

SCORE_COLUMNS = [
    "experience_fit_score",
    "title_company_signal",
    "skill_tier_score",
    "location_fit_score",
    "behavioral_availability_score",
]


def stream_jsonl_extension(jsonl_path: str) -> pd.DataFrame:
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8", buffering=1 << 20) as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            skill_names = {s.get("name") for s in record.get("skills") or [] if s.get("name")}

            has_cv_speech = bool(skill_names & CV_SPEECH_SKILLS)
            has_nlp_ir = bool(skill_names & NLP_IR_SKILLS)
            disq_cv_speech_only = has_cv_speech and not has_nlp_ir

            companies = {
                j.get("company")
                for j in record.get("career_history") or []
                if j.get("company")
            }
            disq_consulting_only = bool(companies) and companies.issubset(CONSULTING_FIRMS)

            matched_ai = [
                s
                for s in record.get("skills") or []
                if s.get("name") in AI_SKILLS_TIER_BC
            ]
            shallow_ai_skill_only = bool(matched_ai) and all(
                (s.get("duration_months") or 0) < 12 for s in matched_ai
            )

            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "disq_cv_speech_only": disq_cv_speech_only,
                    "disq_consulting_only": disq_consulting_only,
                    "shallow_ai_skill_only": shallow_ai_skill_only,
                }
            )
    return pd.DataFrame(rows)


def compute_experience_fit_score(df: pd.DataFrame) -> pd.Series:
    center, sigma = 7.0, 2.5
    return np.exp(-0.5 * ((df["years_of_experience"] - center) / sigma) ** 2)


def compute_title_company_signal(df: pd.DataFrame) -> pd.Series:
    title_score = np.select(
        [df["current_title"].isin(AI_TITLES), df["current_title"].isin(SWE_ADJACENT_TITLES)],
        [1.0, 0.4],
        default=0.0,
    )
    company_score = np.select(
        [
            df["current_company"].isin(CONSULTING_FIRMS),
            df["current_company"].isin(AI_STARTUPS_BIGTECH),
            df["current_company"].isin(PRODUCT_UNICORNS),
        ],
        [0.1, 1.0, 0.7],
        default=0.5,
    )
    return np.clip(0.6 * title_score + 0.4 * company_score, 0.0, 1.0)


def compute_skill_tier_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw = (
        1.0 * df["num_tier_d_skills"]
        + 0.6 * df["num_tier_c_skills"]
        + 0.15 * df["num_tier_b_skills"]
    )
    score = np.clip(raw / 3.0, 0.0, 1.0)
    is_keyword_stuffer = (
        (df["num_tier_b_skills"] >= 5)
        & (df["num_tier_c_skills"] == 0)
        & (df["num_tier_d_skills"] == 0)
    )
    skill_tier_score = np.where(is_keyword_stuffer, score * 0.3, score)
    return pd.Series(skill_tier_score, index=df.index), is_keyword_stuffer


def compute_location_fit_score(df: pd.DataFrame) -> pd.Series:
    city = df["location"].str.split(",").str[0].str.strip()
    willing = df["willing_to_relocate"].astype(bool)
    india = df["country"] == "India"
    tier1_not_jd = city.isin(TIER1_CITIES - {"Pune", "Noida"})

    return pd.Series(
        np.select(
            [
                city.isin({"Pune", "Noida"}),
                tier1_not_jd & willing,
                tier1_not_jd & ~willing,
                india & ~city.isin(TIER1_CITIES) & willing,
                india & ~city.isin(TIER1_CITIES) & ~willing,
                ~india & willing,
            ],
            [1.0, 0.85, 0.55, 0.5, 0.25, 0.3],
            default=0.05,
        ),
        index=df.index,
    )


def compute_behavioral_availability_score(df: pd.DataFrame) -> pd.Series:
    recency_decay = np.exp(-df["days_since_last_active"] / 90.0)
    return (
        0.35 * df["recruiter_response_rate"]
        + 0.25 * df["interview_completion_rate"]
        + 0.25 * recency_decay
        + 0.15 * df["open_to_work_flag"].astype(float)
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["experience_fit_score"] = compute_experience_fit_score(df)
    df["title_company_signal"] = compute_title_company_signal(df)
    skill_tier_score, is_keyword_stuffer = compute_skill_tier_score(df)
    df["skill_tier_score"] = skill_tier_score
    df["is_keyword_stuffer"] = is_keyword_stuffer
    df["location_fit_score"] = compute_location_fit_score(df)
    df["behavioral_availability_score"] = compute_behavioral_availability_score(df)
    df["any_hard_disqualifier"] = (
        df["confirmed_honeypot"] | df["disq_consulting_only"] | df["disq_cv_speech_only"]
    )
    return df


def assert_acceptance_criteria(df: pd.DataFrame) -> None:
    assert len(df) == 100_000, f"expected 100000 rows, got {len(df)}"
    assert df["disq_cv_speech_only"].sum() == 20_293, (
        f"disq_cv_speech_only expected 20293, got {df['disq_cv_speech_only'].sum()}"
    )
    assert df["disq_consulting_only"].sum() == 9_745, (
        f"disq_consulting_only expected 9745, got {df['disq_consulting_only'].sum()}"
    )
    assert df["shallow_ai_skill_only"].sum() == 2_545, (
        f"shallow_ai_skill_only expected 2545, got {df['shallow_ai_skill_only'].sum()}"
    )
    assert df["is_keyword_stuffer"].sum() == 6_539, (
        f"is_keyword_stuffer expected 6539, got {df['is_keyword_stuffer'].sum()}"
    )
    assert df["any_hard_disqualifier"].sum() == 28_191, (
        f"any_hard_disqualifier expected 28191, got {df['any_hard_disqualifier'].sum()}"
    )
    for col in SCORE_COLUMNS:
        null_count = df[col].isna().sum()
        assert null_count == 0, f"{col} has {null_count} nulls"


def print_summary(df: pd.DataFrame) -> None:
    print("\nScore column statistics:")
    print(df[SCORE_COLUMNS].describe().to_string())

    corr_cols = SCORE_COLUMNS + ["confirmed_honeypot"]
    corr = df[corr_cols].astype(float).corr().round(2)
    print("\nCorrelation matrix (5 scores + confirmed_honeypot):")
    print(corr.to_string())


def build_scored_features(input_path: str, jsonl_path: str) -> pd.DataFrame:
    base = pd.read_parquet(input_path)
    extension = stream_jsonl_extension(jsonl_path)
    merged = base.merge(extension, on="candidate_id", how="left")
    assert len(merged) == 100_000, f"merge produced {len(merged)} rows, expected 100000"
    for col in ("disq_cv_speech_only", "disq_consulting_only", "shallow_ai_skill_only"):
        assert merged[col].notna().all(), f"{col} has nulls after merge"
    return engineer_features(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 feature engineering.")
    parser.add_argument(
        "--input",
        default="./data/candidates_features.parquet",
        help="Phase 1 feature parquet",
    )
    parser.add_argument(
        "--jsonl",
        default="./data/candidates.jsonl",
        help="Raw candidates JSONL",
    )
    parser.add_argument(
        "--output",
        default="./data/candidates_scored_features.parquet",
        help="Output scored feature parquet",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = build_scored_features(args.input, args.jsonl)
    elapsed = time.perf_counter() - t0

    assert_acceptance_criteria(df)
    print_summary(df)

    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} rows to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
