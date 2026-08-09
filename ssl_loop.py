from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEED_TRAIN_PATH = "./data/seed_train.parquet"
SEED_VAL_PATH   = "./data/seed_val.parquet"
COMBINED_PATH   = "./data/phase4b_combined_scores.parquet"
OUTPUT_PATH     = "./data/phase4c_ssl_scores.parquet"
LOG_PATH        = "./data/ssl_loop_log.json"

SCORE_COL        = "score"
CANDIDATE_ID_COL = "candidate_id"
RANDOM_SEED      = 42

NEG_SEED_COUNT   = 80    # Match positive seed count → 1:1 ratio
                         # 500 → 6:1 ratio tha → model negatives pe overfit ho gaya tha
NEG_SEED_SCORE   = 0.05  # Score assigned to negative seeds

MAX_ROUNDS       = 8
MIN_NEW_LABELS   = 50    # Stop agar ek round mein itne se kam pseudo-labels mile
PATIENCE         = 8     # Kitne consecutive rounds MAE improve na ho toh stop
                         # 2 → 3: zyada rounds → better convergence

# LightGBM features — sirf tabular, s_text nahi (co-training independence)
TABULAR_FEATURES = [
    "experience_fit_score",
    "title_company_signal",
    "skill_tier_score",
    "location_fit_score",
    "behavioral_availability_score",
]

# Final ensemble weights (same as Phase 4B)
TEXT_WEIGHT    = 0.40
TABULAR_WEIGHT = 0.60

# Agreement thresholds — har round mein thoda relax karo
# Format: (max_diff, high_min, low_max)
ROUND_THRESHOLDS = [
    (0.10, 0.80, 0.20),   # Round 1 — strict
    (0.12, 0.78, 0.22),   # Round 2
    (0.14, 0.75, 0.25),   # Round 3
    (0.15, 0.73, 0.27),   # Round 4
    (0.17, 0.70, 0.30),   # Round 5
    (0.18, 0.68, 0.32),   # Round 6
    (0.20, 0.65, 0.35),   # Round 7
    (0.22, 0.62, 0.38),   # Round 8
]


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      train_df  → labelled seed (starts at 80, grows each round)
      val_df    → fixed validation set (20 candidates, never changes)
      pool_df   → eligible unlabelled candidates (shrinks each round)
    """
    seed_train = pd.read_parquet(SEED_TRAIN_PATH)
    seed_val   = pd.read_parquet(SEED_VAL_PATH)
    combined   = pd.read_parquet(COMBINED_PATH)

    # Merge s_text into seed sets (4B se aaya tha, 4A mein nahi tha)
    text_cols  = [CANDIDATE_ID_COL, "s_text", "s_text_raw", "s_tabular",
                  "view_diff", "view_avg",
                  "agree_high_confidence", "agree_low_confidence",
                  "views_disagree", "any_hard_disqualifier"]
    text_cols  = [c for c in text_cols if c in combined.columns]

    seed_train = seed_train.merge(combined[text_cols], on=CANDIDATE_ID_COL, how="left")
    seed_val   = seed_val.merge(combined[text_cols],   on=CANDIDATE_ID_COL, how="left")

    # Unlabelled pool = eligible candidates NOT in seed sets
    seed_ids  = set(seed_train[CANDIDATE_ID_COL]) | set(seed_val[CANDIDATE_ID_COL])
    eligible  = combined[~combined["any_hard_disqualifier"]].copy()
    pool_df   = eligible[~eligible[CANDIDATE_ID_COL].isin(seed_ids)].copy()

    print(f"✅ Data loaded:")
    print(f"   Labelled train    : {len(seed_train):>7,}")
    print(f"   Labelled val      : {len(seed_val):>7,}")
    print(f"   Unlabelled pool   : {len(pool_df):>7,}")
    print(f"   Total eligible    : {len(eligible):>7,}")

    # Validate features exist
    for col in TABULAR_FEATURES:
        assert col in seed_train.columns, f"Feature '{col}' missing in seed_train"
        assert col in pool_df.columns,    f"Feature '{col}' missing in pool_df"

    assert "s_text" in seed_train.columns, "s_text missing — check Phase 4B ran correctly"

    # ── Negative seed injection ────────────────────────────
    # Eligible pool mein sirf HIGH scorers the — LOW examples nahi the
    # Bottom NEG_SEED_COUNT candidates ko forcefully negative labels do
    # Taaki model "bad candidate" bhi seekhe
    seed_ids_updated = set(seed_train[CANDIDATE_ID_COL]) | set(seed_val[CANDIDATE_ID_COL])
    pool_for_neg     = eligible[~eligible[CANDIDATE_ID_COL].isin(seed_ids_updated)]

    neg_seeds = pool_for_neg.nsmallest(NEG_SEED_COUNT, "composite_score").copy()
    neg_seeds[SCORE_COL] = NEG_SEED_SCORE

    # Ensure all required columns exist in neg_seeds
    required_cols = [CANDIDATE_ID_COL, SCORE_COL] + TABULAR_FEATURES + ["s_text"]
    for col in required_cols:
        if col not in neg_seeds.columns:
            neg_seeds[col] = 0.0

    seed_train = pd.concat(
        [seed_train, neg_seeds[required_cols + ["any_hard_disqualifier"]]],
        ignore_index=True
    )

    # Remove neg_seed candidates from pool (already labelled now)
    neg_seed_ids = set(neg_seeds[CANDIDATE_ID_COL])
    pool_df      = pool_df[~pool_df[CANDIDATE_ID_COL].isin(neg_seed_ids)].copy()

    print(f"\n✅ Negative seeds injected:")
    print(f"   Bottom {NEG_SEED_COUNT} eligible candidates → score = {NEG_SEED_SCORE}")
    print(f"   Updated labelled train : {len(seed_train):>7,}")
    print(f"   Updated unlabelled pool: {len(pool_df):>7,}")

    return seed_train, seed_val, pool_df


# ──────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────
def train_text_calibrator(train_df: pd.DataFrame) -> Pipeline:
    """
    View 1: Polynomial Ridge Regression on s_text
    
    Kyun Ridge (na ki Isotonic)?
    - IsotonicRegression 80 samples pe step function ban jaati hai
      → Bahut saare candidates ko same score deti hai (0.895 band)
    - Ridge (degree=2) smooth curve fit karta hai
      → Better discrimination, no step function collapse
    
    Degree=2 (quadratic) kyun?
    - Linear: agar JD ke liye text similarity aur score mein
              non-linear relationship ho toh miss kar deta
    - Degree=2: slight curve — enough flexibility, no overfit
    - Degree>3: 80-580 samples pe overfit ho jaata
    """
    X = train_df[["s_text"]].values
    y = np.clip(train_df[SCORE_COL].values, 0.0, 1.0)

    model = Pipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=True)),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
    ])
    model.fit(X, y)
    return model


def train_lgbm(train_df: pd.DataFrame, round_num: int) -> LGBMRegressor:
    """
    View 2: LightGBM on tabular features
    n_estimators grows slightly each round — more data = can afford deeper model
    """
    n_est = min(100 + round_num * 20, 300)   # 100 → 260 trees over 8 rounds

    model = LGBMRegressor(
        n_estimators      = n_est,
        learning_rate     = 0.05,
        max_depth         = 4,           # Shallow — only 5 features
        num_leaves        = 15,
        min_child_samples = max(5, len(train_df) // 20),  # Avoid overfit on small data
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = RANDOM_SEED,
        verbose           = -1,
    )

    X = train_df[TABULAR_FEATURES].values
    y = train_df[SCORE_COL].values
    model.fit(X, y)
    return model


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict_both_views(
    pool_df: pd.DataFrame,
    text_model: IsotonicRegression,
    lgbm_model: LGBMRegressor,
) -> pd.DataFrame:
    """
    Predict scores from both views on unlabelled pool.
    Returns pool_df with new prediction columns added.
    """
    pool_df = pool_df.copy()

    # View 1: text calibration (Ridge pipeline needs 2D input)
    pool_df["pred_text"]    = np.clip(
        text_model.predict(pool_df[["s_text"]].values), 0.0, 1.0
    ).astype(np.float32)

    # View 2: LightGBM tabular
    pool_df["pred_tabular"] = np.clip(
        lgbm_model.predict(pool_df[TABULAR_FEATURES].values), 0.0, 1.0
    ).astype(np.float32)

    # Agreement metrics
    pool_df["pred_diff"] = np.abs(
        pool_df["pred_text"] - pool_df["pred_tabular"]
    ).astype(np.float32)
    pool_df["pred_avg"]  = (
        (pool_df["pred_text"] + pool_df["pred_tabular"]) / 2.0
    ).astype(np.float32)

    return pool_df


# ──────────────────────────────────────────────
# AGREEMENT CHECK → PSEUDO-LABELS
# ──────────────────────────────────────────────
def extract_pseudo_labels(
    pool_df: pd.DataFrame,
    round_num: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Co-training agreement rule:
      Both views predict a HIGH or LOW score
      AND their predictions are CLOSE to each other

    Returns:
      pseudo_df  → new pseudo-labelled candidates (add to training)
      remaining  → candidates still unlabelled (pool for next round)
    """
    idx  = min(round_num - 1, len(ROUND_THRESHOLDS) - 1)
    max_diff, high_min, low_max = ROUND_THRESHOLDS[idx]

    diff = pool_df["pred_diff"]
    avg  = pool_df["pred_avg"]

    agree_high = (diff < max_diff) & (avg >= high_min)
    agree_low  = (diff < max_diff) & (avg <= low_max)
    agreed     = agree_high | agree_low

    pseudo_df  = pool_df[agreed].copy()
    remaining  = pool_df[~agreed].copy()

    # Pseudo-label = average of both views
    pseudo_df[SCORE_COL] = pseudo_df["pred_avg"]

    print(f"   Threshold : diff < {max_diff}, avg > {high_min} or < {low_max}")
    print(f"   Agree HIGH: {agree_high.sum():>6,} candidates")
    print(f"   Agree LOW : {agree_low.sum():>6,} candidates")
    print(f"   Total new : {len(pseudo_df):>6,} pseudo-labels")
    print(f"   Remaining : {len(remaining):>6,} still unlabelled")

    return pseudo_df, remaining


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────
def validate(
    val_df: pd.DataFrame,
    text_model: IsotonicRegression,
    lgbm_model: LGBMRegressor,
) -> dict[str, float]:
    """
    Compute MAE on fixed validation set (20 candidates).
    Track text MAE, tabular MAE, and ensemble MAE separately.
    """
    y_true      = val_df[SCORE_COL].values

    pred_text   = np.clip(text_model.predict(val_df[["s_text"]].values), 0.0, 1.0)
    pred_tabular= np.clip(lgbm_model.predict(val_df[TABULAR_FEATURES].values), 0.0, 1.0)
    pred_ensemble = TEXT_WEIGHT * pred_text + TABULAR_WEIGHT * pred_tabular

    return {
        "mae_text"    : float(mean_absolute_error(y_true, pred_text)),
        "mae_tabular" : float(mean_absolute_error(y_true, pred_tabular)),
        "mae_ensemble": float(mean_absolute_error(y_true, pred_ensemble)),
    }


# ──────────────────────────────────────────────
# FINAL SCORING
# ──────────────────────────────────────────────
def compute_final_ssl_scores(
    combined_df: pd.DataFrame,
    text_model: IsotonicRegression,
    lgbm_model: LGBMRegressor,
) -> pd.DataFrame:
    """
    After loop converges, score ALL candidates with final trained models.
    Disqualified candidates get a floor score of 0.01.
    """
    df = combined_df.copy()

    eligible_mask = ~df["any_hard_disqualifier"]

    # Predict on eligible only
    elig = df[eligible_mask].copy()

    pred_text     = np.clip(text_model.predict(elig[["s_text"]].values), 0.0, 1.0)
    pred_tabular  = np.clip(lgbm_model.predict(elig[TABULAR_FEATURES].values), 0.0, 1.0)
    pred_ensemble = (TEXT_WEIGHT * pred_text + TABULAR_WEIGHT * pred_tabular).astype(np.float32)

    df.loc[eligible_mask, "ssl_pred_text"]    = pred_text.astype(np.float32)
    df.loc[eligible_mask, "ssl_pred_tabular"] = pred_tabular.astype(np.float32)
    df.loc[eligible_mask, "ssl_final_score"]  = pred_ensemble

    # Disqualified → floor score
    df.loc[~eligible_mask, "ssl_pred_text"]    = 0.0
    df.loc[~eligible_mask, "ssl_pred_tabular"] = 0.0
    df.loc[~eligible_mask, "ssl_final_score"]  = 0.01

    # Final rank
    df = df.sort_values("ssl_final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ──────────────────────────────────────────────
# MAIN SSL LOOP
# ──────────────────────────────────────────────
def run_ssl_loop(
    seed_train: pd.DataFrame,
    seed_val: pd.DataFrame,
    pool_df: pd.DataFrame,
    max_rounds: int,
    min_new_labels: int,
) -> tuple[IsotonicRegression, LGBMRegressor, list[dict]]:

    labelled_df  = seed_train.copy()
    remaining    = pool_df.copy()
    log          = []
    best_mae     = float("inf")
    patience_ctr = 0

    # Best model checkpoint
    best_text_model = None
    best_lgbm_model = None
    best_round      = 0

    print(f"\n{'='*60}")
    print(f"SSL LOOP STARTING")
    print(f"  Max rounds     : {max_rounds}")
    print(f"  Min new labels : {min_new_labels}")
    print(f"  Patience       : {PATIENCE} rounds")
    print(f"{'='*60}")

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}  |  Labelled set: {len(labelled_df):,} candidates")
        print(f"{'─'*60}")

        # ── Train both models ──────────────────────────────────
        print(f"\n  [1/4] Training models...")
        text_model = train_text_calibrator(labelled_df)
        lgbm_model = train_lgbm(labelled_df, round_num)
        print(f"        Text calibrator  : Ridge (poly=2, {len(labelled_df)} samples)")
        print(f"        LightGBM tabular : {min(100 + round_num * 20, 300)} trees ({len(labelled_df)} samples)")

        # ── Validate ──────────────────────────────────────────
        print(f"\n  [2/4] Validating on {len(seed_val)} held-out candidates...")
        metrics = validate(seed_val, text_model, lgbm_model)
        print(f"        MAE text     : {metrics['mae_text']:.4f}")
        print(f"        MAE tabular  : {metrics['mae_tabular']:.4f}")
        print(f"        MAE ensemble : {metrics['mae_ensemble']:.4f}  ← main metric")

        # Early stopping check
        current_mae = metrics["mae_ensemble"]
        if current_mae < best_mae - 0.001:
            best_mae        = current_mae
            patience_ctr    = 0
            best_text_model = text_model   # ← checkpoint save
            best_lgbm_model = lgbm_model   # ← checkpoint save
            best_round      = round_num
            print(f"        ✅ Best MAE so far! ({best_mae:.4f}) — model checkpoint saved")
        else:
            patience_ctr += 1
            print(f"        ⚠️  No improvement ({patience_ctr}/{PATIENCE} patience)")

        # ── Predict on pool ───────────────────────────────────
        print(f"\n  [3/4] Predicting on {len(remaining):,} unlabelled candidates...")
        pool_with_preds = predict_both_views(remaining, text_model, lgbm_model)

        # ── Agreement check → pseudo-labels ───────────────────
        print(f"\n  [4/4] Agreement check (co-training)...")
        pseudo_df, remaining = extract_pseudo_labels(pool_with_preds, round_num)

        # Log this round
        round_log = {
            "round"           : round_num,
            "labelled_before" : len(labelled_df),
            "new_pseudo_labels": len(pseudo_df),
            "labelled_after"  : len(labelled_df) + len(pseudo_df),
            "remaining_pool"  : len(remaining),
            **metrics,
        }
        log.append(round_log)

        # Add pseudo-labels to training set
        if len(pseudo_df) > 0:
            pseudo_for_train = pseudo_df[[CANDIDATE_ID_COL, SCORE_COL] + TABULAR_FEATURES + ["s_text"]].copy()
            labelled_df = pd.concat([labelled_df, pseudo_for_train], ignore_index=True)

        # ── Stop conditions ───────────────────────────────────
        if len(pseudo_df) < min_new_labels:
            print(f"\n  🛑 STOP: New pseudo-labels ({len(pseudo_df)}) < min threshold ({min_new_labels})")
            break

        if patience_ctr >= PATIENCE:
            print(f"\n  🛑 STOP: MAE not improving for {PATIENCE} consecutive rounds")
            break

        if len(remaining) == 0:
            print(f"\n  🛑 STOP: Unlabelled pool exhausted")
            break

    print(f"\n{'='*60}")
    print(f"SSL LOOP COMPLETE")
    print(f"  Final labelled set  : {len(labelled_df):,} candidates")
    print(f"  Best ensemble MAE   : {best_mae:.4f}  (Round {best_round})")
    print(f"  Rounds completed    : {len(log)}")
    print(f"  Using model from    : Round {best_round} (not Round {len(log)})")
    print(f"  → Pseudo-label drift prevent hoga ✅")
    print(f"{'='*60}")

    return best_text_model, best_lgbm_model, log


# ──────────────────────────────────────────────
# SUMMARY REPORT
# ──────────────────────────────────────────────
def print_round_summary(log: list[dict]) -> None:
    print(f"\n📊 Round-by-Round Summary:")
    print(f"  {'Round':>5}  {'Labelled':>9}  {'+New':>6}  {'MAE_txt':>8}  {'MAE_tab':>8}  {'MAE_ens':>8}")
    print(f"  {'─'*60}")
    for r in log:
        print(
            f"  {r['round']:>5}  "
            f"{r['labelled_before']:>9,}  "
            f"{r['new_pseudo_labels']:>+6,}  "
            f"{r['mae_text']:>8.4f}  "
            f"{r['mae_tabular']:>8.4f}  "
            f"{r['mae_ensemble']:>8.4f}"
        )


def print_top_candidates(df: pd.DataFrame, n: int = 15) -> None:
    eligible = df[~df["any_hard_disqualifier"]]
    print(f"\n🏆 TOP {n} CANDIDATES (final SSL ranking):")
    print(f"  {'Rank':>4}  {'candidate_id':<20}  {'ssl_score':>9}  {'text':>6}  {'tab':>6}  title")
    print(f"  {'─'*80}")
    for _, row in eligible.head(n).iterrows():
        title = str(row.get("current_title", ""))[:30]
        print(
            f"  {int(row['rank']):>4}  "
            f"{row[CANDIDATE_ID_COL]:<20}  "
            f"{row['ssl_final_score']:>9.4f}  "
            f"{row['ssl_pred_text']:>6.3f}  "
            f"{row['ssl_pred_tabular']:>6.3f}  "
            f"{title}"
        )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4C — SSL Loop")
    parser.add_argument("--seed_train",     default=SEED_TRAIN_PATH)
    parser.add_argument("--seed_val",       default=SEED_VAL_PATH)
    parser.add_argument("--combined",       default=COMBINED_PATH)
    parser.add_argument("--output",         default=OUTPUT_PATH)
    parser.add_argument("--log",            default=LOG_PATH)
    parser.add_argument("--max_rounds",     type=int, default=MAX_ROUNDS)
    parser.add_argument("--min_new_labels", type=int, default=MIN_NEW_LABELS)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4C: SSL Loop (Co-Training + Self-Training)")
    print("=" * 60)

    # Load data
    seed_train, seed_val, pool_df = load_data()

    # Run SSL loop
    text_model, lgbm_model, log = run_ssl_loop(
        seed_train, seed_val, pool_df,
        max_rounds     = args.max_rounds,
        min_new_labels = args.min_new_labels,
    )

    # Summary
    print_round_summary(log)

    # Final scoring on all 100,000 candidates
    print(f"\n⚙️  Computing final scores on all 100,000 candidates...")
    combined_df = pd.read_parquet(args.combined)
    final_df    = compute_final_ssl_scores(combined_df, text_model, lgbm_model)

    # Preview
    print_top_candidates(final_df, n=15)

    # Save
    out_cols = [
        CANDIDATE_ID_COL,
        "rank",
        "ssl_final_score",
        "ssl_pred_text",
        "ssl_pred_tabular",
        "composite_score",     # Phase 4B score (baseline comparison)
        "s_text",
        "s_tabular",
        "any_hard_disqualifier",
        "confirmed_honeypot",
        "current_title",
        "current_company",
        "years_of_experience",
    ]
    out_cols = [c for c in out_cols if c in final_df.columns]
    final_df[out_cols].to_parquet(args.output, index=False)

    # Save log
    Path(args.log).write_text(json.dumps(log, indent=2))

    print(f"\n💾 Saved:")
    print(f"   {args.output}  ({len(final_df):,} rows)")
    print(f"   {args.log}  (round-by-round metrics)")
    print(f"\n✅ Phase 4C complete → Next: phase_5_submit.py")


if __name__ == "__main__":
    main()
