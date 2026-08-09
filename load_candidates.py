from __future__ import annotations

import json
import warnings
from collections import Counter
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

TODAY = date(2026, 6, 24)

EDUCATION_TIER_RANK = {
    "tier_1": 1,
    "tier_2": 2,
    "tier_3": 3,
    "tier_4": 4,
    "unknown": 5,
}

ADVANCED_DEGREE_MARKERS = ("M.", "Ph.D", "MBA")


def parse_date(value: str) -> date:
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def skill_tier_for_freq(freq: int) -> str:
    if freq > 8000:
        return "A"
    if freq >= 3000:
        return "B"
    if freq >= 800:
        return "C"
    return "D"


def build_skill_tier_map(skill_freq: Counter[str]) -> dict[str, str]:
    return {name: skill_tier_for_freq(freq) for name, freq in skill_freq.items()}


def print_skill_tier_table(skill_freq: Counter[str]) -> None:
    rows = sorted(skill_freq.items(), key=lambda x: (-x[1], x[0]))
    print("skill_name\tfreq\ttier")
    for name, freq in rows:
        print(f"{name}\t{freq}\t{skill_tier_for_freq(freq)}")


def stream_records(path: str):
    with open(path, encoding="utf-8", buffering=1 << 20) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def count_skill_frequencies(path: str) -> Counter[str]:
    freq: Counter[str] = Counter()
    for record in stream_records(path):
        for skill in record.get("skills") or []:
            name = skill.get("name")
            if name:
                freq[name] += 1
    return freq


def _identify_current_job(
    career_history: list[dict[str, Any]],
    candidate_id: str = "",
    warn: bool = False,
) -> dict[str, Any] | None:
    if not career_history:
        return None

    current_entries = [j for j in career_history if j.get("is_current") is True]
    if len(current_entries) == 1:
        return current_entries[0]

    if warn:
        if len(current_entries) > 1:
            warnings.warn(
                f"{candidate_id}: multiple is_current jobs ({len(current_entries)}); "
                "falling back to latest start_date",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{candidate_id}: no is_current job; falling back to latest start_date",
                stacklevel=2,
            )

    def start_key(job: dict[str, Any]) -> date:
        start = job.get("start_date")
        return parse_date(start) if start else date.min

    return max(career_history, key=start_key)


def _highest_education_tier(education: list[dict[str, Any]]) -> str | None:
    if not education:
        return None
    best_rank = max(EDUCATION_TIER_RANK.values())
    best_tier = "unknown"
    for entry in education:
        tier = entry.get("tier") or "unknown"
        rank = EDUCATION_TIER_RANK.get(tier, EDUCATION_TIER_RANK["unknown"])
        if rank < best_rank:
            best_rank = rank
            best_tier = tier
    return best_tier


def _has_advanced_degree(education: list[dict[str, Any]]) -> bool:
    for entry in education:
        degree = entry.get("degree") or ""
        if any(marker in degree for marker in ADVANCED_DEGREE_MARKERS):
            return True
    return False


def flatten_record(
    record: dict[str, Any],
    skill_tier_map: dict[str, str],
    today: date = TODAY,
) -> dict[str, Any]:
    profile = record.get("profile") or {}
    career_history = record.get("career_history") or []
    education = record.get("education") or []
    skills = record.get("skills") or []
    signals = record.get("redrob_signals") or {}

    candidate_id = record["candidate_id"]
    location = profile.get("location") or ""
    country = profile.get("country") or ""
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""

    current_job = _identify_current_job(career_history, candidate_id, warn=False)
    current_job_title = current_job.get("title") if current_job else None
    current_job_company = current_job.get("company") if current_job else None
    current_job_industry = current_job.get("industry") if current_job else None

    num_jobs = len(career_history)
    total_tenure_months = sum(j.get("duration_months") or 0 for j in career_history)
    max_single_job_duration = (
        max((j.get("duration_months") or 0 for j in career_history), default=0)
    )
    industries = {
        j.get("industry")
        for j in career_history
        if j.get("industry")
    }

    edu_start_years = [e.get("start_year") for e in education if e.get("start_year") is not None]
    edu_end_years = [e.get("end_year") for e in education if e.get("end_year") is not None]

    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    num_expert = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            num_expert += 1
        tier = skill_tier_map.get(skill.get("name", ""), "D")
        tier_counts[tier] += 1

    salary = signals.get("expected_salary_range_inr_lpa") or {}
    salary_min = salary.get("min")
    salary_max = salary.get("max")
    raw_salary_inverted_flag = False
    if salary_min is not None and salary_max is not None and salary_min > salary_max:
        raw_salary_inverted_flag = True
        salary_min, salary_max = min(salary_min, salary_max), max(salary_min, salary_max)

    signup_raw = signals.get("signup_date")
    last_active_raw = signals.get("last_active_date")
    signup_date = parse_date(signup_raw) if signup_raw else None
    last_active_date = parse_date(last_active_raw) if last_active_raw else None
    raw_last_active_before_signup_flag = False
    if signup_date and last_active_date and last_active_date < signup_date:
        raw_last_active_before_signup_flag = True
        last_active_date = signup_date

    days_since_last_active = (
        (today - last_active_date).days if last_active_date is not None else np.nan
    )

    assessment_scores = signals.get("skill_assessment_scores") or {}
    avg_assessment = (
        sum(assessment_scores.values()) / len(assessment_scores)
        if assessment_scores
        else np.nan
    )

    github_score = signals.get("github_activity_score")
    offer_rate = signals.get("offer_acceptance_rate")

    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "years_of_experience": profile.get("years_of_experience"),
        "location": location,
        "country": country,
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "current_industry": profile.get("current_industry"),
        "headline_len": len(headline),
        "summary_len": len(summary),
        "is_india": country == "India",
        "is_pune_or_noida": ("Pune" in location) or ("Noida" in location),
        "num_jobs": num_jobs,
        "total_tenure_months": total_tenure_months,
        "max_single_job_duration_months": max_single_job_duration,
        "num_distinct_industries_in_history": len(industries),
        "current_job_title": current_job_title,
        "current_job_company": current_job_company,
        "current_job_industry": current_job_industry,
        "title_company_consistency_flag": (
            profile.get("current_title") == current_job_title
            and profile.get("current_company") == current_job_company
        ),
        "num_education_entries": len(education),
        "highest_tier": _highest_education_tier(education),
        "has_advanced_degree": _has_advanced_degree(education),
        "earliest_start_year": min(edu_start_years) if edu_start_years else np.nan,
        "latest_end_year": max(edu_end_years) if edu_end_years else np.nan,
        "num_skills": len(skills),
        "num_expert_skills": num_expert,
        "num_tier_a_skills": tier_counts["A"],
        "num_tier_b_skills": tier_counts["B"],
        "num_tier_c_skills": tier_counts["C"],
        "num_tier_d_skills": tier_counts["D"],
        "tier_c_to_b_ratio": tier_counts["C"] / max(tier_counts["B"], 1),
        "num_certifications": len(record.get("certifications") or []),
        "num_languages": len(record.get("languages") or []),
        "profile_completeness_score": signals.get("profile_completeness_score"),
        "signup_date": signup_date,
        "last_active_date": last_active_date,
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "profile_views_received_30d": signals.get("profile_views_received_30d"),
        "applications_submitted_30d": signals.get("applications_submitted_30d"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "avg_response_time_hours": signals.get("avg_response_time_hours"),
        "num_skill_assessments": len(assessment_scores),
        "avg_skill_assessment_score": avg_assessment,
        "connection_count": signals.get("connection_count"),
        "endorsements_received": signals.get("endorsements_received"),
        "notice_period_days": signals.get("notice_period_days"),
        "expected_salary_min_lpa": salary_min,
        "expected_salary_max_lpa": salary_max,
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "github_activity_score": github_score,
        "has_github": github_score != -1 if github_score is not None else False,
        "search_appearance_30d": signals.get("search_appearance_30d"),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
        "offer_acceptance_rate": offer_rate,
        "has_offer_history": offer_rate != -1 if offer_rate is not None else False,
        "verified_email": signals.get("verified_email"),
        "verified_phone": signals.get("verified_phone"),
        "linkedin_connected": signals.get("linkedin_connected"),
        "raw_salary_inverted_flag": raw_salary_inverted_flag,
        "raw_last_active_before_signup_flag": raw_last_active_before_signup_flag,
        "days_since_last_active": days_since_last_active,
    }
    return row


def load_candidates(
    path: str,
    today: date = TODAY,
    print_tiers: bool = True,
) -> pd.DataFrame:
    skill_freq = count_skill_frequencies(path)
    skill_tier_map = build_skill_tier_map(skill_freq)
    if print_tiers:
        print_skill_tier_table(skill_freq)

    rows = [
        flatten_record(record, skill_tier_map, today=today)
        for record in stream_records(path)
    ]
    return pd.DataFrame(rows)
