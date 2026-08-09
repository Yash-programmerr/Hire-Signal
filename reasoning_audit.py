import json
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

SUBMISSION_CSV = "your_participant_id.csv"   # tumhari final top-100 file
CANDIDATES_JSONL = "./data/candidates.jsonl"
SAMPLE_SIZE = 10
RANDOM_SEED = 42


def load_all_skill_names(jsonl_path, n_scan=5000):
    names = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_scan:
                break
            rec = json.loads(line)
            for s in rec["skills"]:
                names.add(s["name"])
    return names


def get_candidate_skills(jsonl_path, candidate_ids):
    wanted = set(candidate_ids)
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["candidate_id"] in wanted:
                out[rec["candidate_id"]] = {
                    "skills": {s["name"] for s in rec["skills"]},
                    "years_of_experience": rec["profile"]["years_of_experience"],
                    "current_title": rec["profile"]["current_title"],
                }
                wanted.discard(rec["candidate_id"])
                if not wanted:
                    break
    return out


def check_hallucination(reasoning, actual_skills, current_title, all_skill_vocab):
    mentioned = set()
    for name in all_skill_vocab:
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, reasoning, re.IGNORECASE):
            if name.lower() in current_title.lower():
                continue  # part of job-title text, not a skill claim
            mentioned.add(name)
    return mentioned - actual_skills


def check_variation(reasonings):
    sims = []
    for i in range(len(reasonings)):
        for j in range(i + 1, len(reasonings)):
            sims.append(SequenceMatcher(None, reasonings[i], reasonings[j]).ratio())
    return sum(sims) / len(sims) if sims else 0.0


def check_specific_facts(reasoning):
    return bool(re.search(r"\d", reasoning))


def main():
    df = pd.read_csv(SUBMISSION_CSV)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED).sort_values("rank")

    all_skill_vocab = load_all_skill_names(CANDIDATES_JSONL)
    cand_info = get_candidate_skills(CANDIDATES_JSONL, sample["candidate_id"].tolist())

    print(f"REASONING SELF-AUDIT — {len(sample)} sampled rows (same method Stage 4 uses)\n")

    issues = []
    reasonings = sample["reasoning"].fillna("").tolist()

    empty_count = sum(1 for r in reasonings if not r.strip())
    if empty_count:
        issues.append(f"{empty_count}/{len(reasonings)} reasonings are empty")
    if len(set(reasonings)) == 1:
        issues.append("ALL sampled reasonings are identical")

    avg_sim = check_variation(reasonings)
    if avg_sim > 0.6:
        issues.append(f"Average pairwise similarity is {avg_sim:.2f} (>0.6) — looks templated")

    for _, row in sample.iterrows():
        cid, rank, reasoning = row["candidate_id"], row["rank"], str(row["reasoning"])
        info = cand_info.get(cid)
        print(f"Rank {rank:>3} | {cid} | {info['current_title'] if info else '?'} "
              f"({info['years_of_experience'] if info else '?'} yrs)")
        print(f"   Reasoning: {reasoning}")

        if info:
            hallucinated = check_hallucination(reasoning, info["skills"], info["current_title"], all_skill_vocab)
            if hallucinated:
                print(f"   HALLUCINATION: mentions {hallucinated} — not in candidate's actual skills")
                issues.append(f"{cid} (rank {rank}): hallucinated skill(s) {hallucinated}")
            else:
                print("   No hallucinated skills detected")

        if not check_specific_facts(reasoning):
            issues.append(f"{cid} (rank {rank}): no specific numeric fact in reasoning")

        print("   --> Manually judge: JD connection? Honest concerns acknowledged? "
              "Tone matches rank?\n")

    print(f"Average pairwise similarity: {avg_sim:.2f}\n")
    if issues:
        print("AUTOMATED ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("No automated issues. Still manually read the 10 rows for JD-connection / "
              "honest-concerns / rank-tone — those need human judgment.")


if __name__ == "__main__":
    main()