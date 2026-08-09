import argparse
import json
import sys
import time

from scoring import score_candidate

TOP_N = 100


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("[1/3] Scoring all candidates...")
    rows = []
    with open(args.candidates, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            score, reasoning, flags = score_candidate(rec)
            rows.append({
                "candidate_id": rec["candidate_id"],
                "score": score,
                "reasoning": reasoning,
                "confirmed_honeypot": flags["honeypot"],
            })

    print(f"      {len(rows)} candidates scored")

    print("[2/3] Sorting and selecting top 100...")
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)

    top = df.head(TOP_N).copy()
    top["rank"] = range(1, TOP_N + 1)
    final = top[["candidate_id", "rank", "score", "reasoning"]]

    print("[3/3] Validating before write...")
    errors = []
    if len(final) != TOP_N:
        errors.append(f"Row count {len(final)} != {TOP_N}")
    if final["candidate_id"].duplicated().any():
        errors.append("Duplicate candidate_ids")
    if final["score"].nunique() == 1:
        errors.append("All scores identical")
    scores_list = final["score"].tolist()
    if any(scores_list[i] < scores_list[i+1] for i in range(len(scores_list)-1)):
        errors.append("Score not non-increasing with rank")
    if not final["score"].between(0, 1).all():
        errors.append("Scores outside [0,1]")
    if final["reasoning"].isna().any():
        errors.append("Empty reasoning rows")

    hp = top["confirmed_honeypot"].sum()
    print(f"      Honeypots in top {TOP_N}: {hp}")
    if hp > 10:
        errors.append(f"{hp} honeypots exceed 10% threshold")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    final.to_csv(args.out, index=False)
    elapsed = time.time() - t_start
    print(f"\nSaved {len(final)} rows -> {args.out}")
    print(f"Total time: {elapsed:.1f}s")
    print("\nTop 10:")
    print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()