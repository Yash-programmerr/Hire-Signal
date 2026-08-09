import pandas as pd
import sys

INPUT_FILE = "Sheild_Clan.csv"        # tumhari latest 100K-row file
OUTPUT_FILE = "your_participant_id.csv"   # <-- apna actual registered participant ID daalo yahan

df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df)} rows")

required_cols = ["candidate_id", "rank", "score", "reasoning"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"ERROR: missing columns {missing}")
    sys.exit(1)

# --- Sort: score DESC, tie-break candidate_id ASC (spec-approved method) ---
df_sorted = df.sort_values(
    ["score", "candidate_id"],
    ascending=[False, True]
).reset_index(drop=True)

# --- Take EXACTLY top 100, reassign rank 1-100 sequentially ---
top100 = df_sorted.head(100).copy()
top100["rank"] = range(1, 101)
top100 = top100[required_cols]   # exact column order

# ============================================================
# Pre-flight checks — every rule from submission_spec.docx Section 3 & 6
# ============================================================
errors = []

if len(top100) != 100:
    errors.append(f"Row count is {len(top100)}, must be exactly 100")

if sorted(top100["rank"].tolist()) != list(range(1, 101)):
    errors.append("Ranks are not exactly 1-100, each appearing once")

if top100["rank"].min() == 0:
    errors.append("Rank starts at 0 instead of 1")

if top100["candidate_id"].duplicated().any():
    errors.append(f"Duplicate candidate_ids: {top100['candidate_id'][top100['candidate_id'].duplicated()].tolist()}")

if top100["score"].nunique() == 1:
    errors.append("All scores are identical — model isn't differentiating")

scores = top100["score"].tolist()
if any(scores[i] < scores[i+1] for i in range(len(scores)-1)):
    errors.append("Score is NOT non-increasing with rank (some later rank scores higher)")

if top100["score"].between(0, 1).all() == False:
    errors.append("Some scores are outside [0, 1]")

if top100["reasoning"].isna().any() or (top100["reasoning"].str.strip() == "").any():
    errors.append("Some rows have empty reasoning")

# Optional but important: honeypot check (need confirmed_honeypot from your scored parquet)
try:
    scored = pd.read_parquet("data/candidates_scored_features.parquet")
    hp_ids = set(scored.loc[scored["confirmed_honeypot"], "candidate_id"])
    hp_in_top100 = top100["candidate_id"].isin(hp_ids).sum()
    if hp_in_top100 > 10:
        errors.append(f"{hp_in_top100} honeypots in top 100 — exceeds 10% disqualification threshold")
    else:
        print(f"Honeypot check: {hp_in_top100}/100 in top 100 (threshold: 10)")
except FileNotFoundError:
    print("Skipping honeypot check — candidates_scored_features.parquet not found here")

if errors:
    print("\n FAILED pre-flight checks:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)

print("\n All pre-flight checks passed.")
top100.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(top100)} rows -> {OUTPUT_FILE}")
print("\nTop 10 preview:")
print(top100.head(10).to_string(index=False))