from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR        = Path("./data/")
SCORED_FEATURES = DATA_DIR / "candidates_scored_features.parquet"
SIM_SCORES_PATH = DATA_DIR / "semantic_similarity_scores.npy"
EMBED_IDS_PATH  = DATA_DIR / "embedding_candidate_ids.npy"
OUTPUT_PATH     = DATA_DIR / "phase4b_combined_scores.parquet"

# Tabular score weights — reflect JD priorities
# JD: Senior NLP/Search Ranking Engineer at Redrob
# → Skills aur title sabse important hain
TABULAR_WEIGHTS = {
    "skill_tier_score":              0.40,  # was 0.30
    "experience_fit_score":          0.25,  # was 0.20
    "behavioral_availability_score": 0.20,  # was 0.15
    "location_fit_score":            0.15,  # was 0.10
}

TITLE_GATE_FLOOR = 0.30

# Final ensemble weights
TEXT_WEIGHT    = 0.40   # Semantic understanding
TABULAR_WEIGHT = 0.60   # Objective signals

# Honeypot penalty multiplier (0.05 = score almost zero)
HONEYPOT_PENALTY = 0.05

# Agreement thresholds for SSL loop (Round 0 conservative settings)
AGREE_DIFF_MAX   = 0.10   # Max allowed gap between text & tabular
AGREE_HIGH_MIN   = 0.80   # Min avg score to be "confidently good"
AGREE_LOW_MAX    = 0.20   # Max avg score to be "confidently bad"
DISAGREE_MIN     = 0.25   # Diff >= this → actively disagree


# ──────────────────────────────────────────────
# STEP 1: Load Text Scores (View 1)
# ──────────────────────────────────────────────
def load_text_scores(sim_path: Path, ids_path: Path) -> pd.DataFrame:
    """
    Load precomputed cosine similarity scores from embed_candidates.py
    Returns DataFrame with candidate_id and raw cosine similarity
    """
    if not sim_path.exists():
        raise FileNotFoundError(
            f"{sim_path} nahi mili!\n"
            "Pehle embed_candidates.py run karo:\n"
            "  python embed_candidates.py --jsonl ./data/candidates.jsonl"
        )

    sim_scores    = np.load(sim_path).astype(np.float32)
    candidate_ids = np.load(ids_path, allow_pickle=True).astype(str)

    assert len(sim_scores) == len(candidate_ids), "Mismatch between scores and IDs"
    assert len(sim_scores) == 100_000, f"Expected 100000 scores, got {len(sim_scores)}"

    text_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "s_text_raw":   sim_scores,         # Raw cosine similarity [-1, 1]
    })

    print(f"✅ Text scores loaded: {len(text_df):,} candidates")
    print(f"   Raw cosine sim — min: {sim_scores.min():.4f}, max: {sim_scores.max():.4f}")
    print(f"                    mean: {sim_scores.mean():.4f}, std: {sim_scores.std():.4f}")

    return text_df


# ──────────────────────────────────────────────
# STEP 2: Normalize Text Scores to [0, 1]
# ──────────────────────────────────────────────
def normalize_text_scores(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity range is [-1, 1].
    Tabular scores are already [0, 1].
    Min-max normalize so both are on same scale.

    Why min-max (not standard normalization)?
    → We want hard 0–1 boundaries, not a bell curve.
      Tabular scores are bounded [0,1] so text should match.
    """
    s     = text_df["s_text_raw"]
    s_min = s.min()
    s_max = s.max()

    text_df["s_text"] = ((s - s_min) / (s_max - s_min)).astype(np.float32)

    print(f"\n✅ Text scores normalized to [0, 1]")
    print(f"   Normalization: (x - {s_min:.4f}) / {(s_max - s_min):.4f}")
    print(f"   Normalized — min: {text_df['s_text'].min():.4f}, max: {text_df['s_text'].max():.4f}")
    print(f"                 mean: {text_df['s_text'].mean():.4f}, std: {text_df['s_text'].std():.4f}")

    return text_df


# ──────────────────────────────────────────────
# STEP 3: Load Tabular Features (View 2)
# ──────────────────────────────────────────────
def load_tabular_features(features_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    assert len(df) == 100_000, f"Expected 100000 rows, got {len(df)}"

    # Verify all 5 score columns exist
    for col in TABULAR_WEIGHTS:
        assert col in df.columns, f"Column '{col}' not found in features parquet"

    print(f"\n✅ Tabular features loaded: {len(df):,} candidates")
    print(f"   Hard disqualifiers : {df['any_hard_disqualifier'].sum():,}")
    print(f"   Confirmed honeypots: {df['confirmed_honeypot'].sum():,}")
    print(f"   Eligible candidates: {(~df['any_hard_disqualifier']).sum():,}")

    return df


# ──────────────────────────────────────────────
# STEP 4: Composite Tabular Score
# ──────────────────────────────────────────────
def compute_composite_tabular_score(df: pd.DataFrame) -> pd.Series:
    """
    Weighted sum of 4 tabular dimensions → base score,
    then GATED by title_company_signal (not averaged in).

    Kyun gate, average nahi?
    - Averaging: ek Graphic Designer ke 11 (wrong-domain) skills uske
      0.2 title score ko 75% weight se overpower kar dete the.
    - Gating: title_company_signal ab ek multiplier hai — agar title/company
      JD se match nahi karta, to chahe skill/experience kitna bhi high ho,
      final score proportionally crush hota hai.
    """
    s_tabular = np.zeros(len(df), dtype=np.float32)

    print(f"\n📊 Tabular score component contributions (pre-gate):")
    print(f"   {'Component':<35} {'Weight':>6}  {'Avg Score':>9}  {'Avg Contrib':>11}")
    print(f"   {'-'*65}")

    for col, weight in TABULAR_WEIGHTS.items():
        contribution = weight * df[col].values
        s_tabular += contribution
        print(f"   {col:<35} {weight:>6.2f}  {df[col].mean():>9.4f}  {contribution.mean():>11.4f}")

    s_tabular = np.clip(s_tabular, 0.0, 1.0)

    # ── Gate by title_company_signal ──
    title_gate = TITLE_GATE_FLOOR + (1 - TITLE_GATE_FLOOR) * df["title_company_signal"].values
    s_tabular_gated = np.clip(s_tabular * title_gate, 0.0, 1.0)

    print(f"   {'-'*65}")
    print(f"   {'PRE-GATE COMPOSITE':<35}  {'':>6}  {s_tabular.mean():>9.4f}")
    print(f"   {'TITLE GATE (avg multiplier)':<35}  {'':>6}  {title_gate.mean():>9.4f}")
    print(f"   {'POST-GATE S_tabular':<35}  {'':>6}  {s_tabular_gated.mean():>9.4f}")
    print(f"\n   S_tabular range : {s_tabular_gated.min():.4f} – {s_tabular_gated.max():.4f}")
    print(f"   S_tabular std   : {s_tabular_gated.std():.4f}")

    return pd.Series(s_tabular_gated, index=df.index, name="s_tabular")


# ──────────────────────────────────────────────
# STEP 5: Merge Both Views
# ──────────────────────────────────────────────
def merge_views(tabular_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    merged = tabular_df.merge(
        text_df[["candidate_id", "s_text", "s_text_raw"]],
        on="candidate_id",
        how="left"
    )

    missing = merged["s_text"].isna().sum()
    if missing > 0:
        print(f"\n⚠️  WARNING: {missing} candidates missing text scores → filled with 0.0")
        merged["s_text"]     = merged["s_text"].fillna(0.0).astype(np.float32)
        merged["s_text_raw"] = merged["s_text_raw"].fillna(0.0).astype(np.float32)

    print(f"\n✅ Both views merged: {len(merged):,} total candidates")
    return merged


# ──────────────────────────────────────────────
# STEP 6: Final Ensemble Score
# ──────────────────────────────────────────────
def compute_final_score(df: pd.DataFrame) -> pd.Series:
    """
    Final score = weighted combination of text + tabular views

    Honeypots & hard disqualifiers:
    → Score multiply by HONEYPOT_PENALTY (effectively sends to bottom)
    → Still keep a non-zero score to avoid submission edge cases
    """
    base_score = (
        TEXT_WEIGHT    * df["s_text"] +
        TABULAR_WEIGHT * df["s_tabular"]
    ).astype(np.float32)

    # Apply penalty to disqualified candidates
    final_score = np.where(
        df["any_hard_disqualifier"],
        base_score * HONEYPOT_PENALTY,
        base_score
    ).astype(np.float32)

    eligible_mask = ~df["any_hard_disqualifier"]
    print(f"\n📊 Final composite score:")
    print(f"   All candidates    — mean: {final_score.mean():.4f}, std: {final_score.std():.4f}")
    print(f"   Eligible only     — mean: {final_score[eligible_mask].mean():.4f}, "
          f"std: {final_score[eligible_mask].std():.4f}")
    print(f"   Disqualified      — mean: {final_score[~eligible_mask].mean():.4f} "
          f"(after {HONEYPOT_PENALTY}x penalty)")
    print(f"\n   Ensemble weights: TEXT={TEXT_WEIGHT:.0%} | TABULAR={TABULAR_WEIGHT:.0%}")

    return pd.Series(final_score, index=df.index, name="composite_score")


# ──────────────────────────────────────────────
# STEP 7: Agreement Flags for SSL Loop
# ──────────────────────────────────────────────
def compute_agreement_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag candidates by how much text vs tabular views agree.
    These flags drive the SSL co-training loop in Phase 4C.

    agree_high → Both views say "good candidate" → add as positive pseudo-label
    agree_low  → Both views say "bad candidate"  → add as negative pseudo-label
    disagree   → Views conflict strongly          → skip this round
    middle     → Scores in ambiguous zone         → skip (too uncertain)
    """
    diff = np.abs(df["s_text"] - df["s_tabular"]).astype(np.float32)
    avg  = ((df["s_text"] + df["s_tabular"]) / 2.0).astype(np.float32)

    eligible = ~df["any_hard_disqualifier"]

    agree_high = eligible & (diff < AGREE_DIFF_MAX) & (avg > AGREE_HIGH_MIN)
    agree_low  = eligible & (diff < AGREE_DIFF_MAX) & (avg < AGREE_LOW_MAX)
    disagree   = eligible & (diff >= DISAGREE_MIN)
    middle     = eligible & ~agree_high & ~agree_low & ~disagree

    df["view_diff"]              = diff
    df["view_avg"]               = avg
    df["agree_high_confidence"]  = agree_high   # Positive pseudo-label candidates
    df["agree_low_confidence"]   = agree_low    # Negative pseudo-label candidates
    df["views_disagree"]         = disagree     # Skip these in SSL
    df["middle_zone"]            = middle       # Ambiguous — skip for now

    total_eligible = eligible.sum()
    print(f"\n📊 Agreement Analysis (Round 0 — before SSL loop):")
    print(f"   Total eligible candidates : {total_eligible:>7,}")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   ✅ Agree HIGH (confident +ve) : {agree_high.sum():>7,} "
          f"({agree_high.sum()/total_eligible:.1%})")
    print(f"   ✅ Agree LOW  (confident -ve) : {agree_low.sum():>7,} "
          f"({agree_low.sum()/total_eligible:.1%})")
    print(f"   ❌ Disagree   (diff≥{DISAGREE_MIN})   : {disagree.sum():>7,} "
          f"({disagree.sum()/total_eligible:.1%})")
    print(f"   ⚠️  Middle zone (ambiguous)   : {middle.sum():>7,} "
          f"({middle.sum()/total_eligible:.1%})")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   SSL Round 0 pseudo-labels available: {agree_high.sum() + agree_low.sum():>7,}")

    return df


# ──────────────────────────────────────────────
# STEP 8: Sanity Check — Top & Bottom Preview
# ──────────────────────────────────────────────
def print_top_bottom_preview(df: pd.DataFrame, n: int = 10) -> None:
    eligible = df[~df["any_hard_disqualifier"]].copy()

    display_cols = ["candidate_id", "current_title", "composite_score",
                    "s_text", "s_tabular", "view_diff"]

    print(f"\n{'='*75}")
    print(f"TOP {n} CANDIDATES (eligible only)")
    print(f"{'='*75}")
    top = eligible.nlargest(n, "composite_score")[display_cols]
    print(top.to_string(index=False))

    print(f"\n{'='*75}")
    print(f"BOTTOM {n} CANDIDATES (eligible only)")
    print(f"{'='*75}")
    bottom = eligible.nsmallest(n, "composite_score")[display_cols]
    print(bottom.to_string(index=False))

    # Score percentile thresholds
    print(f"\n📊 Score Percentiles (eligible candidates):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(eligible["composite_score"], p)
        print(f"   P{p:>2} : {val:.4f}")


# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────
def save_output(df: pd.DataFrame, out_path: Path) -> None:
    # Columns to keep in output (everything SSL loop + ranking will need)
    keep_cols = [
        "candidate_id",
        # Final scores
        "composite_score",
        "s_text",
        "s_text_raw",
        "s_tabular",
        # Individual tabular components
        "experience_fit_score",
        "title_company_signal",
        "skill_tier_score",
        "location_fit_score",
        "behavioral_availability_score",
        # Agreement flags for SSL
        "view_diff",
        "view_avg",
        "agree_high_confidence",
        "agree_low_confidence",
        "views_disagree",
        "middle_zone",
        # Disqualifier flags
        "confirmed_honeypot",
        "any_hard_disqualifier",
        "disq_cv_speech_only",
        "disq_consulting_only",
        # Useful metadata
        "current_title",
        "current_company",
        "years_of_experience",
        "is_keyword_stuffer",
        "shallow_ai_skill_only",
    ]

    # Only keep columns that exist in df
    out_cols = [c for c in keep_cols if c in df.columns]
    out_df   = df[out_cols]

    out_df.to_parquet(out_path, index=False)

    print(f"\n💾 Saved → {out_path}")
    print(f"   Shape : {len(out_df):,} rows × {len(out_cols)} columns")
    print(f"   Size  : {out_path.stat().st_size / 1e6:.1f} MB")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4B — Combine Both Views")
    parser.add_argument("--features",     default=str(SCORED_FEATURES))
    parser.add_argument("--sim_scores",   default=str(SIM_SCORES_PATH))
    parser.add_argument("--embed_ids",    default=str(EMBED_IDS_PATH))
    parser.add_argument("--output",       default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4B: Combining Views → Composite Score")
    print("=" * 60)

    # Step 1 & 2: Text View
    text_df = load_text_scores(Path(args.sim_scores), Path(args.embed_ids))
    text_df = normalize_text_scores(text_df)

    # Step 3: Tabular View
    tabular_df = load_tabular_features(Path(args.features))

    # Step 4: Composite tabular score
    tabular_df["s_tabular"] = compute_composite_tabular_score(tabular_df)

    # Step 5: Merge
    df = merge_views(tabular_df, text_df)

    # Step 6: Final score
    df["composite_score"] = compute_final_score(df)

    # Step 7: Agreement flags
    df = compute_agreement_flags(df)

    # Step 8: Preview
    print_top_bottom_preview(df, n=10)

    # Save
    save_output(df, Path(args.output))

    print(f"\n✅ Phase 4B complete → {args.output}")
    print("   Next step → Run phase_4c_ssl_loop.py")


if __name__ == "__main__":
    main()
