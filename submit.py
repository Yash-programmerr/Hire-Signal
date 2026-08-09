from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SSL_SCORES_PATH  = "./data/phase4c_ssl_scores.parquet"
OUTPUT_CSV       = "./submission.csv"
VALIDATOR_PATH   = "./validate_submission.py"
CANDIDATE_ID_COL = "candidate_id"

# Submission spec ke hisaab se columns
SUBMIT_COLS = ["candidate_id", "rank", "score", "reasoning"]


# ──────────────────────────────────────────────
# STEP 1: Load SSL Scores
# ──────────────────────────────────────────────
def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert CANDIDATE_ID_COL in df.columns, "candidate_id column missing"
    assert "ssl_final_score" in df.columns, "ssl_final_score missing — run Phase 4C first"
    assert len(df) == 100_000, f"Expected 100,000 rows, got {len(df)}"

    print(f"✅ Scores loaded: {len(df):,} total candidates")
    print(f"   Eligible     : {(~df['any_hard_disqualifier']).sum():,}")
    print(f"   Disqualified : {df['any_hard_disqualifier'].sum():,}")
    print(f"   Honeypots    : {df['confirmed_honeypot'].sum():,}")
    return df


# ──────────────────────────────────────────────
# STEP 2: Generate Reasoning
# ──────────────────────────────────────────────
def generate_reasoning(row: pd.Series) -> str:
    """
    Har candidate ke liye ek short reasoning string generate karo.
    Submission spec mein reasoning optional hai lekin judges appreciate karte hain.
    """
    if row.get("any_hard_disqualifier", False):
        if row.get("confirmed_honeypot", False):
            return "Disqualified: confirmed honeypot candidate"
        elif row.get("disq_cv_speech_only", False):
            return "Disqualified: no relevant AI/NLP experience"
        elif row.get("disq_consulting_only", False):
            return "Disqualified: only consulting firm experience, no product ML"
        return "Disqualified: hard filter"

    # Build reasoning from scores
    parts = []

    ssl   = row.get("ssl_final_score", 0)
    text  = row.get("ssl_pred_text", row.get("s_text", 0))
    tab   = row.get("ssl_pred_tabular", row.get("s_tabular", 0))

    if ssl >= 0.85:
        parts.append("Strong overall fit")
    elif ssl >= 0.70:
        parts.append("Good overall fit")
    elif ssl >= 0.50:
        parts.append("Moderate fit")
    else:
        parts.append("Below-average fit")

    if text >= 0.90:
        parts.append("excellent semantic alignment with JD")
    elif text >= 0.75:
        parts.append("good semantic alignment with JD")

    if tab >= 0.85:
        parts.append("strong tabular signals (skills/title/experience)")
    elif tab >= 0.70:
        parts.append("good tabular signals")
    elif tab < 0.40:
        parts.append("weak tabular signals")

    yoe = row.get("years_of_experience", None)
    if yoe is not None:
        if 5 <= yoe <= 9:
            parts.append(f"{int(yoe)}yr experience (ideal range)")
        elif yoe < 5:
            parts.append(f"{int(yoe)}yr experience (below preferred)")
        else:
            parts.append(f"{int(yoe)}yr experience (over-qualified)")

    return "; ".join(parts) if parts else "Scored by SSL ensemble model"


# ──────────────────────────────────────────────
# STEP 3: Build Submission DataFrame
# ──────────────────────────────────────────────
def build_submission(df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """
    Eligible → ssl_final_score se sort (descending), tie-break candidate_id ascending
    Disqualified → bottom pe (score = 0.01, flat), tie-break candidate_id ascending
    """
    eligible = df[~df["any_hard_disqualifier"]].copy()
    disq     = df[df["any_hard_disqualifier"]].copy()

    # Sort eligible by ssl_final_score descending, tie-break by candidate_id ascending
    eligible = eligible.sort_values(
        by=["ssl_final_score", CANDIDATE_ID_COL],
        ascending=[False, True]
    ).reset_index(drop=True)
    eligible["rank"]  = eligible.index + 1
    eligible["score"] = eligible["ssl_final_score"].round(6)

    # Disqualified — flat score, ranked after eligible, same tie-break rule
    disq = disq.sort_values(
        by=["ssl_final_score", CANDIDATE_ID_COL],
        ascending=[False, True]
    ).reset_index(drop=True)
    disq["rank"]  = disq.index + 1 + len(eligible)
    disq["score"] = 0.01

    # Combine
    combined = pd.concat([eligible, disq], ignore_index=True)

    # Generate reasoning
    print(f"\n⚙️  Generating reasoning strings...")
    combined["reasoning"] = combined.apply(generate_reasoning, axis=1)

    # Final submission columns
    submission = combined[[CANDIDATE_ID_COL, "rank", "score", "reasoning"]].copy()

    # Optional: top_k only
    if top_k is not None:
        print(f"   Trimming to top {top_k} candidates only")
        submission = submission[submission["rank"] <= top_k]

    print(f"\n✅ Submission built:")
    print(f"   Total rows         : {len(submission):,}")
    print(f"   Eligible ranked    : {len(eligible):,}")
    print(f"   Disqualified       : {len(disq):,}")
    print(f"   Score range        : {submission['score'].min():.4f} – {submission['score'].max():.4f}")
    print(f"   Rank range         : {submission['rank'].min()} – {submission['rank'].max()}")

    return submission


# ──────────────────────────────────────────────
# STEP 4: Quality Checks
# ──────────────────────────────────────────────
def quality_checks(submission: pd.DataFrame, df: pd.DataFrame) -> None:
    print(f"\n🔍 Quality Checks:")

    # 1. No duplicate candidate IDs
    dupes = submission[CANDIDATE_ID_COL].duplicated().sum()
    status = "✅" if dupes == 0 else "❌"
    print(f"   {status} Duplicate candidate IDs : {dupes}")

    # 2. Ranks are unique and consecutive
    rank_dupes = submission["rank"].duplicated().sum()
    status = "✅" if rank_dupes == 0 else "❌"
    print(f"   {status} Duplicate ranks          : {rank_dupes}")

    # 3. Scores in [0, 1]
    bad_scores = (~submission["score"].between(0, 1)).sum()
    status = "✅" if bad_scores == 0 else "❌"
    print(f"   {status} Scores outside [0,1]     : {bad_scores}")

    # 4. No honeypots in top 100
    top100_ids      = set(submission[submission["rank"] <= 100][CANDIDATE_ID_COL])
    honeypot_ids    = set(df[df["confirmed_honeypot"]][CANDIDATE_ID_COL])
    hp_in_top100    = len(top100_ids & honeypot_ids)
    status = "✅" if hp_in_top100 == 0 else "❌"
    print(f"   {status} Honeypots in top 100     : {hp_in_top100}")

    # 5. Disqualified not in top 500
    top500_ids      = set(submission[submission["rank"] <= 500][CANDIDATE_ID_COL])
    disq_ids        = set(df[df["any_hard_disqualifier"]][CANDIDATE_ID_COL])
    disq_in_top500  = len(top500_ids & disq_ids)
    status = "✅" if disq_in_top500 == 0 else "⚠️ "
    print(f"   {status} Disqualified in top 500  : {disq_in_top500}")

    # 6. Top 20 title distribution
    top20 = submission[submission["rank"] <= 20]
    top20_ids = top20[CANDIDATE_ID_COL].tolist()
    top20_data = df[df[CANDIDATE_ID_COL].isin(top20_ids)][
        [CANDIDATE_ID_COL, "current_title", "ssl_final_score",
         "ssl_pred_text", "ssl_pred_tabular"]
    ].sort_values("ssl_final_score", ascending=False)

    print(f"\n📋 Top 20 Candidate Breakdown:")
    print(f"   {'Rank':>4}  {'Score':>6}  {'Text':>5}  {'Tab':>5}  Title")
    print(f"   {'─'*70}")
    for i, (_, row) in enumerate(top20_data.iterrows(), 1):
        title = str(row.get("current_title", ""))[:35]
        print(
            f"   {i:>4}  "
            f"{row['ssl_final_score']:>6.4f}  "
            f"{row.get('ssl_pred_text', 0):>5.3f}  "
            f"{row.get('ssl_pred_tabular', 0):>5.3f}  "
            f"{title}"
        )


# ──────────────────────────────────────────────
# STEP 5: Save + Validate
# ──────────────────────────────────────────────
def save_and_validate(submission: pd.DataFrame, out_path: str) -> None:
    # Sort by rank ASCENDING (1 = best) to get the top 100, not the bottom 100.
    # Rank already has candidate_id-ascending tie-breaking baked in from
    # build_submission(), so we don't need to re-sort by candidate_id here —
    # doing so would undo the correct tie-break order.
    df = submission.sort_values(by="rank", ascending=True).head(100)
    df.to_csv(out_path, index=False)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"\n💾 Saved → {out_path}  ({len(df):,} rows, {size_kb:.1f} KB)")

    # Run official validator if available
    validator = Path(VALIDATOR_PATH)
    if validator.exists():
        print(f"\n🔄 Running official validator: {VALIDATOR_PATH}")
        result = subprocess.run(
            ["python", VALIDATOR_PATH, out_path],   # positional arg, not --submission
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        if result.returncode == 0:
            print("✅ Official validation passed!")
        else:
            print(f"❌ Validator returned code {result.returncode}")
    else:
        print(f"\n⚠️  Validator not found at {VALIDATOR_PATH} — skipping official validation")
        print(f"   Manually run: python validate_submission.py {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 — Submission Generator")
    parser.add_argument("--scores",  default=SSL_SCORES_PATH)
    parser.add_argument("--output",  default=OUTPUT_CSV)
    parser.add_argument("--top_k",   type=int, default=None,
                        help="Submit only top K candidates (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 5: Final Submission Generator")
    print("=" * 60)

    df         = load_scores(args.scores)
    submission = build_submission(df, top_k=args.top_k)
    quality_checks(submission, df)
    save_and_validate(submission, args.output)

    print(f"\n{'='*60}")
    print(f"✅ Submission ready → {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()