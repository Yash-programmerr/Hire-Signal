from __future__ import annotations

import argparse
import time
from datetime import date

import pandas as pd

from honeypot_flags import combine_flags, print_flag_summary
from load_candidates import (
    TODAY,
    build_skill_tier_map,
    count_skill_frequencies,
    flatten_record,
    print_skill_tier_table,
    stream_records,
)


def build_feature_table(
    input_path: str,
    today: date = TODAY,
    print_tiers: bool = True,
) -> pd.DataFrame:
    skill_freq = count_skill_frequencies(input_path)
    skill_tier_map = build_skill_tier_map(skill_freq)
    if print_tiers:
        print_skill_tier_table(skill_freq)

    rows: list[dict] = []
    flag_rows: list[dict] = []
    for record in stream_records(input_path):
        rows.append(flatten_record(record, skill_tier_map, today=today))
        flag_rows.append(combine_flags(record, today=today))

    df = pd.DataFrame(rows)
    flags_df = pd.DataFrame(flag_rows)
    return pd.concat([df, flags_df], axis=1)


def assert_acceptance_criteria(df: pd.DataFrame) -> None:
    assert len(df) == 100_000, f"expected 100000 rows, got {len(df)}"
    assert df["candidate_id"].is_unique, "candidate_id must be unique"
    assert df["flag_expert_zero_duration"].sum() == 21, (
        f"flag_expert_zero_duration expected 21, got {df['flag_expert_zero_duration'].sum()}"
    )
    assert df["flag_duration_mismatch"].sum() == 35, (
        f"flag_duration_mismatch expected 35, got {df['flag_duration_mismatch'].sum()}"
    )
    assert df["confirmed_honeypot"].sum() == 56, (
        f"confirmed_honeypot expected 56, got {df['confirmed_honeypot'].sum()}"
    )
    for col in ("candidate_id", "years_of_experience", "current_title"):
        null_count = df[col].isna().sum()
        assert null_count == 0, f"{col} has {null_count} nulls"


def print_profile_summary(df: pd.DataFrame) -> None:
    print("\n=== Feature table profile ===")
    print(f"rows: {len(df):,}")
    print("\ndtype breakdown:")
    print(df.dtypes.value_counts())

    rule_cols = [
        "flag_expert_zero_duration",
        "flag_duration_mismatch",
        "flag_no_github_many_experts",
        "flag_education_impossible",
        "flag_title_company_mismatch",
        "flag_skill_duration_exceeds_tenure",
        "flag_overlapping_jobs",
        "suspicious_yoe_mismatch",
    ]
    print("\nPer-rule counts:")
    for col in rule_cols:
        print(f"  {col}: {int(df[col].sum())}")
    print(f"  confirmed_honeypot: {int(df['confirmed_honeypot'].sum())}")

    print("\nsuspicious_flag_count distribution:")
    print(df["suspicious_flag_count"].value_counts().sort_index().to_string())

    stat_cols = [
        "years_of_experience",
        "total_tenure_months",
        "recruiter_response_rate",
        "days_since_last_active",
    ]
    print("\nBasic stats:")
    print(df[stat_cols].describe().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate feature parquet.")
    parser.add_argument(
        "--input",
        default="./data/candidates.jsonl",
        help="Path to candidates JSONL",
    )
    parser.add_argument(
        "--output",
        default="./data/candidates_features.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Reference date YYYY-MM-DD (default 2026-06-24)",
    )
    args = parser.parse_args()

    today = date.fromisoformat(args.today) if args.today else TODAY

    t0 = time.perf_counter()
    df = build_feature_table(args.input, today=today)
    elapsed = time.perf_counter() - t0

    assert_acceptance_criteria(df)
    print_profile_summary(df)

    flag_rows = df[
        [
            "flag_expert_zero_duration",
            "flag_duration_mismatch",
            "flag_no_github_many_experts",
            "flag_education_impossible",
            "flag_title_company_mismatch",
            "flag_skill_duration_exceeds_tenure",
            "flag_overlapping_jobs",
            "suspicious_yoe_mismatch",
            "confirmed_honeypot",
        ]
    ].to_dict(orient="records")
    print_flag_summary(flag_rows)

    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} rows to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
