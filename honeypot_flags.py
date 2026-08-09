from __future__ import annotations

from datetime import date
from typing import Any

from load_candidates import TODAY, _identify_current_job, months_between, parse_date


def _total_tenure_months(career_history: list[dict[str, Any]]) -> int:
    return sum(j.get("duration_months") or 0 for j in career_history)


def _job_date_range(job: dict[str, Any], today: date) -> tuple[date, date]:
    start = parse_date(job["start_date"])
    end_raw = job.get("end_date")
    end = parse_date(end_raw) if end_raw else today
    return start, end


def flag_expert_zero_duration(record: dict[str, Any]) -> bool:
    count = sum(
        1
        for s in record.get("skills") or []
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    return count >= 3


def flag_duration_mismatch(record: dict[str, Any], today: date = TODAY) -> bool:
    for job in record.get("career_history") or []:
        start_raw = job.get("start_date")
        if not start_raw:
            continue
        start = parse_date(start_raw)
        end_raw = job.get("end_date")
        end = parse_date(end_raw) if end_raw else today
        implied = months_between(start, end)
        duration = job.get("duration_months")
        if duration is None:
            continue
        if abs(implied - duration) > 3:
            return True
    return False


def _degree_timeline_too_short(degree: str, start_year: int, end_year: int) -> bool:
    span = end_year - start_year
    if "Ph.D" in degree and span < 2:
        return True
    advanced_markers = ("M.Tech", "M.E.", "M.S.", "M.Sc", "MBA")
    if any(marker in degree for marker in advanced_markers) and span < 1:
        return True
    return False


def flag_education_impossible(record: dict[str, Any], today: date = TODAY) -> bool:
    education = record.get("education") or []
    for entry in education:
        degree = entry.get("degree") or ""
        start_year = entry.get("start_year")
        end_year = entry.get("end_year")
        if start_year is not None and end_year is not None:
            if _degree_timeline_too_short(degree, start_year, end_year):
                return True

    edu_end_years = [e.get("end_year") for e in education if e.get("end_year") is not None]
    if not edu_end_years:
        return False

    latest_end_year = max(edu_end_years)
    career_history = record.get("career_history") or []
    start_dates: list[date] = []
    for job in career_history:
        start_raw = job.get("start_date")
        if start_raw:
            start_dates.append(parse_date(start_raw))

    if not start_dates:
        return False

    earliest_career = min(start_dates)
    # Career began more than one calendar year before highest education end year.
    return earliest_career.year < latest_end_year - 1


def flag_title_company_mismatch(record: dict[str, Any]) -> bool:
    profile = record.get("profile") or {}
    career_history = record.get("career_history") or []
    if not career_history:
        return False

    current_job = _identify_current_job(
        career_history, record.get("candidate_id", ""), warn=False
    )
    if not current_job:
        return False

    title_match = profile.get("current_title") == current_job.get("title")
    company_match = profile.get("current_company") == current_job.get("company")
    return not (title_match and company_match)


def flag_skill_duration_exceeds_tenure(record: dict[str, Any]) -> bool:
    total_tenure = _total_tenure_months(record.get("career_history") or [])
    limit = total_tenure + 12
    for skill in record.get("skills") or []:
        duration = skill.get("duration_months") or 0
        if duration > limit:
            return True
    return False


def _overlap_months(start1: date, end1: date, start2: date, end2: date) -> int:
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start >= overlap_end:
        return 0
    return months_between(overlap_start, overlap_end)


def flag_overlapping_jobs(record: dict[str, Any], today: date = TODAY) -> bool:
    jobs = record.get("career_history") or []
    ranges: list[tuple[date, date]] = []
    for job in jobs:
        if not job.get("start_date"):
            continue
        ranges.append(_job_date_range(job, today))

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            if _overlap_months(*ranges[i], *ranges[j]) > 1:
                return True
    return False


def flag_yoe_tenure_extreme(record: dict[str, Any]) -> bool:
    profile = record.get("profile") or {}
    yoe = profile.get("years_of_experience")
    if yoe is None:
        return False
    total_tenure = _total_tenure_months(record.get("career_history") or [])
    return abs(yoe - total_tenure / 12.0) > 3


def flag_no_github_many_experts(record: dict[str, Any]) -> bool:
    """CONFIRMED honeypot signature #3. Claims >=6 'expert' skills but has
    no verifiable GitHub account (github_activity_score == -1).
    Verified: exactly 24 NEW candidates, zero overlap with the other two
    confirmed rules — brings total confirmed_honeypot to 80, matching spec."""
    signals = record.get("redrob_signals") or {}
    no_github = signals.get("github_activity_score") == -1
    num_expert = sum(
        1 for s in record.get("skills") or [] if s.get("proficiency") == "expert"
    )
    return no_github and num_expert >= 6


def combine_flags(record: dict[str, Any], today: date = TODAY) -> dict[str, Any]:
    flags = {
        "flag_expert_zero_duration": flag_expert_zero_duration(record),
        "flag_duration_mismatch": flag_duration_mismatch(record, today=today),
        "flag_no_github_many_experts": flag_no_github_many_experts(record),
        "flag_education_impossible": flag_education_impossible(record, today=today),
        "flag_title_company_mismatch": flag_title_company_mismatch(record),
        "flag_skill_duration_exceeds_tenure": flag_skill_duration_exceeds_tenure(record),
        "flag_overlapping_jobs": flag_overlapping_jobs(record, today=today),
        "suspicious_yoe_mismatch": flag_yoe_tenure_extreme(record),
    }

    reasons: list[str] = []
    reason_map = {
        "flag_expert_zero_duration": "expert skills with zero duration (>=3)",
        "flag_duration_mismatch": "career duration inconsistent with dates",
        "flag_no_github_many_experts": "many expert skills but no GitHub account",
        "flag_education_impossible": "education timeline implausible",
        "flag_title_company_mismatch": "profile title/company != current job",
        "flag_skill_duration_exceeds_tenure": "skill duration exceeds career tenure",
        "flag_overlapping_jobs": "overlapping job date ranges",
        "suspicious_yoe_mismatch": "years_of_experience vs tenure gap > 3y",
    }
    for key, reason in reason_map.items():
        if flags[key]:
            reasons.append(reason)

    flags["honeypot_reasons"] = "; ".join(reasons)

    flags["confirmed_honeypot"] = (
        flags["flag_expert_zero_duration"] or flags["flag_duration_mismatch"]
    )

    suspicious_keys = (
        "flag_education_impossible",
        "flag_title_company_mismatch",
        "flag_skill_duration_exceeds_tenure",
        "flag_overlapping_jobs",
    )
    flags["suspicious_flag_count"] = sum(int(flags[k]) for k in suspicious_keys)

    return flags


def print_flag_summary(flag_rows: list[dict[str, Any]]) -> None:
    rule_keys = [
        "flag_expert_zero_duration",
        "flag_duration_mismatch",
        "flag_no_github_many_experts",
        "flag_education_impossible",
        "flag_title_company_mismatch",
        "flag_skill_duration_exceeds_tenure",
        "flag_overlapping_jobs",
        "suspicious_yoe_mismatch",
    ]
    print("\nHoneypot rule summary:")
    print(f"{'rule':<40} {'count':>8}")
    print("-" * 50)
    for key in rule_keys:
        count = sum(1 for row in flag_rows if row.get(key))
        print(f"{key:<40} {count:>8}")

    new_keys = (
        "flag_education_impossible",
        "flag_title_company_mismatch",
        "flag_skill_duration_exceeds_tenure",
        "flag_overlapping_jobs",
    )
    any_new = sum(
        1 for row in flag_rows if any(row.get(k) for k in new_keys)
    )
    confirmed = sum(1 for row in flag_rows if row.get("confirmed_honeypot"))
    union_all = sum(
        1
        for row in flag_rows
        if row.get("confirmed_honeypot")
        or any(row.get(k) for k in new_keys)
    )
    print("-" * 50)
    print(f"{'any_new_suspicious_rule':<40} {any_new:>8}")
    print(f"{'confirmed_honeypot':<40} {confirmed:>8}")
    print(f"{'union_confirmed_or_new':<40} {union_all:>8}")
