from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_SUBMISSION  = "./sample_submission.csv"
SCORED_FEATURES    = "./data/candidates_scored_features.parquet"
OUTPUT_DIR         = "./data/"

TOTAL_SEED_LABELS  = None  # Auto-detected from CSV (don't hardcode)
TRAIN_RATIO        = 0.80  # 80% train, 20% val
RANDOM_SEED        = 42

SCORE_COL          = "score"          # column name in sample_submission.csv
CANDIDATE_ID_COL   = "candidate_id"


# ──────────────────────────────────────────────
# STEP 1: Load & Validate Seed Labels
# ──────────────────────────────────────────────
def load_seed_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic validation — no hardcoded row count
    assert CANDIDATE_ID_COL in df.columns, f"'{CANDIDATE_ID_COL}' column missing in submission CSV"
    assert SCORE_COL in df.columns,        f"'{SCORE_COL}' column missing in submission CSV"
    assert len(df) > 0,                    "Submission CSV is empty!"
    assert df[SCORE_COL].between(0, 1).all(), (
        f"Some scores outside [0,1]: min={df[SCORE_COL].min():.3f}, max={df[SCORE_COL].max():.3f}"
    )

    n_total    = len(df)
    n_dupes    = n_total - df[CANDIDATE_ID_COL].nunique()
    if n_dupes > 0:
        print(f"⚠️  WARNING: {n_dupes} duplicate candidate IDs found — dropping duplicates")
        df = df.drop_duplicates(subset=CANDIDATE_ID_COL, keep="first")

    print(f"✅ Seed labels loaded: {len(df)} candidates")
    print(f"   Score range : {df[SCORE_COL].min():.3f}  –  {df[SCORE_COL].max():.3f}")
    print(f"   Score mean  : {df[SCORE_COL].mean():.3f}")
    print(f"   Score std   : {df[SCORE_COL].std():.3f}")

    # Show distribution in buckets
    buckets = pd.cut(df[SCORE_COL], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    print(f"\n   Score distribution:")
    for bucket, count in buckets.value_counts().sort_index().items():
        bar = "█" * int(count)
        print(f"     {bucket}: {bar} ({count})")

    return df


# ──────────────────────────────────────────────
# STEP 2: Merge with Tabular Features
# ──────────────────────────────────────────────
def merge_with_features(labels: pd.DataFrame, features_path: str) -> pd.DataFrame:
    # Load only necessary columns to save memory
    feature_cols = [
        CANDIDATE_ID_COL,
        "experience_fit_score",
        "title_company_signal",
        "skill_tier_score",
        "location_fit_score",
        "behavioral_availability_score",
        "confirmed_honeypot",
        "any_hard_disqualifier",
        "current_title",
        "current_company",
        "years_of_experience",
    ]

    features = pd.read_parquet(features_path, columns=feature_cols)
    merged   = labels.merge(features, on=CANDIDATE_ID_COL, how="left")

    # Check for candidates not found in features
    missing = merged["experience_fit_score"].isna().sum()
    if missing > 0:
        print(f"\n⚠️  WARNING: {missing} candidates not found in features parquet")
        missing_ids = merged.loc[merged["experience_fit_score"].isna(), CANDIDATE_ID_COL].tolist()
        print(f"   Missing IDs: {missing_ids}")

    # Check if any labelled candidate is a honeypot (should not happen, but verify)
    honeypots_in_labels = merged["confirmed_honeypot"].sum()
    if honeypots_in_labels > 0:
        print(f"\n⚠️  WARNING: {honeypots_in_labels} labelled candidates are confirmed honeypots!")
        print("   These will be kept in train set but flagged.")

    print(f"\n✅ Merged successfully: {len(merged)} labelled candidates with features")
    return merged


# ──────────────────────────────────────────────
# STEP 3: Stratified Train / Validation Split
# ──────────────────────────────────────────────
def stratified_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split — dynamic based on actual CSV size.
    Sort by score, then take every Nth candidate for validation
    so both sets have uniform score distribution.

    Why not random? With small samples, random split
    can give very skewed distributions by chance.
    """
    n_total = len(df)
    n_val   = max(1, round(n_total * (1 - train_ratio)))   # at least 1 val candidate
    n_train = n_total - n_val

    df_sorted = df.sort_values(SCORE_COL).reset_index(drop=True)

    # Pick val candidates evenly spaced across score range
    step        = n_total // n_val
    val_indices = list(range(step // 2, n_total, step))[:n_val]  # centered in each bucket
    train_indices = [i for i in range(n_total) if i not in val_indices]

    train_df = df_sorted.iloc[train_indices].reset_index(drop=True)
    val_df   = df_sorted.iloc[val_indices].reset_index(drop=True)

    print(f"\n📊 Train / Validation Split (ratio={train_ratio:.0%}/{1-train_ratio:.0%}):")
    print(f"   Total seed labels : {n_total}")
    print(f"   Train → {len(train_df)} candidates")
    print(f"     Score mean : {train_df[SCORE_COL].mean():.3f}")
    print(f"     Score std  : {train_df[SCORE_COL].std():.3f}")
    print(f"     Score range: {train_df[SCORE_COL].min():.3f} – {train_df[SCORE_COL].max():.3f}")
    print(f"\n   Val   → {len(val_df)} candidates")
    print(f"     Score mean : {val_df[SCORE_COL].mean():.3f}")
    print(f"     Score std  : {val_df[SCORE_COL].std():.3f}")
    print(f"     Score range: {val_df[SCORE_COL].min():.3f} – {val_df[SCORE_COL].max():.3f}")

    # Show val candidates (small set, good to inspect)
    print(f"\n   Validation candidates (for reference):")
    print(f"   {'candidate_id':<20} {'score':>6}  title")
    print(f"   {'-'*65}")
    for _, row in val_df.iterrows():
        title = str(row.get("current_title", "N/A"))[:35]
        print(f"   {row[CANDIDATE_ID_COL]:<20} {row[SCORE_COL]:>6.3f}  {title}")

    return train_df, val_df


# ──────────────────────────────────────────────
# STEP 4: Save Outputs
# ──────────────────────────────────────────────
def save_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "seed_train.parquet"
    val_path   = out / "seed_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path,     index=False)

    print(f"\n💾 Saved:")
    print(f"   {train_path}  ({len(train_df)} rows)")
    print(f"   {val_path}    ({len(val_df)} rows)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4A — Seed Label Preparation")
    parser.add_argument("--submission", default=SAMPLE_SUBMISSION)
    parser.add_argument("--features",   default=SCORED_FEATURES)
    parser.add_argument("--out_dir",    default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4A: Seed Labels Preparation")
    print("=" * 60)

    labels   = load_seed_labels(args.submission)
    merged   = merge_with_features(labels, args.features)
    train_df, val_df = stratified_split(merged, train_ratio=TRAIN_RATIO)
    save_outputs(train_df, val_df, args.out_dir)

    print("\n✅ Phase 4A complete — seed_train.parquet & seed_val.parquet ready")
    print("   Next step → Run phase_4b_combine_views.py")


if __name__ == "__main__":
    main()
