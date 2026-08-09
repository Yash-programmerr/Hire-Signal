import json
import numpy as np
from datetime import date

TODAY = date(2026, 6, 29)

NLP_IR_SKILLS = {
    "Hugging Face Transformers","LangChain","Information Retrieval","LLMs","Recommendation Systems",
    "Semantic Search","Sentence Transformers","Embeddings","Vector Search","Prompt Engineering",
    "Pinecone","FAISS","RAG","Fine-tuning LLMs","QLoRA","pgvector","Weaviate","Milvus",
    "Learning to Rank","BM25","TensorFlow","Qdrant","Python","PyTorch","PEFT","LoRA","NLP",
    "Machine Learning","Deep Learning","Haystack","Elasticsearch","LlamaIndex","scikit-learn",
    "OpenSearch","Information Retrieval Systems","Search Backend","Text Encoders",
    "Vector Representations","Content Matching","Model Adaptation","Ranking Systems",
    "Search & Discovery","Workflow Orchestration","Search Infrastructure","Indexing Algorithms",
    "Open-source ML libraries","Natural Language Processing","Document Processing"
}

CV_SPEECH_SKILLS = {
    "YOLO","GANs","OpenCV","ASR","Image Classification","Computer Vision",
    "Speech Recognition","CNN","Diffusion Models","TTS","Object Detection"
}

TIER_D_SKILLS = {
    "Information Retrieval Systems","Search Backend","Text Encoders","Vector Representations",
    "Content Matching","Model Adaptation","Ranking Systems","Search & Discovery",
    "Workflow Orchestration","Search Infrastructure","Indexing Algorithms",
    "Open-source ML libraries","Natural Language Processing","Document Processing"
}

CONSULTING_FIRMS = {
    "TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini",
    "HCL","Tech Mahindra","Mphasis","Mindtree"
}

AI_TITLES = {
    "ML Engineer","AI Research Engineer","Data Scientist","Senior Software Engineer (ML)",
    "Computer Vision Engineer","Junior ML Engineer","AI Specialist","Recommendation Systems Engineer",
    "Machine Learning Engineer","Applied ML Engineer","Search Engineer","AI Engineer",
    "Senior Data Scientist","NLP Engineer","Senior NLP Engineer","Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer","Senior AI Engineer","Senior Applied Scientist","Lead AI Engineer"
}

SWE_ADJACENT_TITLES = {
    "Software Engineer","Full Stack Developer","Cloud Engineer","Java Developer",
    ".NET Developer","DevOps Engineer","Mobile Developer","Frontend Engineer","QA Engineer",
    "Analytics Engineer","Data Engineer","Data Analyst","Backend Engineer",
    "Senior Data Engineer","Senior Software Engineer"
}

AI_STARTUPS_BIGTECH = {
    "Genpact AI","Sarvam AI","Aganitha","Rephrase.ai","Niramai","Glance","Haptik","Wysa",
    "Krutrim","Saarthi.ai","Verloop.io","Mad Street Den","Yellow.ai","Locobuzz","Observe.AI",
    "Meta","Google","Netflix","Amazon","Microsoft","Salesforce","LinkedIn","Apple","Adobe","Uber"
}

PRODUCT_UNICORNS = {
    "Swiggy","CRED","Razorpay","Zomato","Flipkart","Meesho","InMobi","Nykaa","Zoho","Freshworks",
    "Vedantu","Ola","Paytm","BYJU'S","upGrad","PolicyBazaar","Dream11","PharmEasy","PhonePe","Unacademy"
}

TIER1_CITIES = {"Bangalore","Mumbai","Delhi","Hyderabad","Chennai","Gurgaon","Pune","Noida"}
ASPIRATIONAL_PHRASES = ["interested in transitioning", "self-directed"]


def parse_date(s):
    if not s:
        return TODAY
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d").date()


def months_between(sd, ed):
    return (ed.year - sd.year) * 12 + (ed.month - sd.month)


# ── Honeypot detection ──────────────────────────────────────────────────────

def is_honeypot(rec):
    skills = rec["skills"]
    career = rec["career_history"]
    signals = rec["redrob_signals"]

    # Rule 1: ≥3 expert skills with duration_months=0
    n = sum(1 for s in skills
            if s.get("proficiency") == "expert" and s.get("duration_months") == 0)
    if n >= 3:
        return True

    # Rule 2: duration_months vs date-math mismatch >3 months
    for j in career:
        sd = parse_date(j["start_date"])
        ed = parse_date(j["end_date"]) if j.get("end_date") else TODAY
        implied = months_between(sd, ed)
        if abs(implied - j.get("duration_months", 0)) > 3:
            return True

    # Rule 3: no GitHub + ≥6 expert skills
    no_github = signals["github_activity_score"] == -1
    num_expert = sum(1 for s in skills if s.get("proficiency") == "expert")
    if no_github and num_expert >= 6:
        return True

    return False


# ── Disqualifier checks ─────────────────────────────────────────────────────

def is_consulting_only(rec):
    companies = {j["company"] for j in rec["career_history"]}
    return bool(companies) and companies.issubset(CONSULTING_FIRMS)


def is_cv_speech_only(rec):
    names = {s["name"] for s in rec["skills"]}
    return bool(names & CV_SPEECH_SKILLS) and not bool(names & NLP_IR_SKILLS)


def is_aspirational_pivot(rec):
    s = (rec["profile"].get("summary") or "").lower()
    return all(p in s for p in ASPIRATIONAL_PHRASES)


# ── Feature scores ──────────────────────────────────────────────────────────

def experience_fit(yoe):
    return float(np.exp(-0.5 * ((yoe - 7.0) / 2.5) ** 2))


def skill_tier_score(skills):
    num_d = sum(1 for s in skills if s["name"] in TIER_D_SKILLS)
    num_c = sum(1 for s in skills if s["name"] in (NLP_IR_SKILLS - TIER_D_SKILLS))
    num_b = sum(1 for s in skills if s["name"] in CV_SPEECH_SKILLS)
    raw = 1.0 * num_d + 0.6 * num_c + 0.15 * num_b
    score = min(raw / 3.0, 1.0)
    # Keyword stuffer penalty
    if num_b >= 5 and num_c == 0 and num_d == 0:
        score *= 0.3
    return score


def title_company_signal(title, company):
    if title in AI_TITLES:
        t = 1.0
    elif title in SWE_ADJACENT_TITLES:
        t = 0.4
    else:
        t = 0.0   # HR Manager, Product Manager, etc. = zero

    if company in CONSULTING_FIRMS:
        c = 0.1
    elif company in AI_STARTUPS_BIGTECH:
        c = 1.0
    elif company in PRODUCT_UNICORNS:
        c = 0.7
    else:
        c = 0.5

    return min(0.6 * t + 0.4 * c, 1.0)


def location_fit(location, country, willing_to_relocate):
    city = location.split(",")[0].strip()
    if city in {"Pune", "Noida"}:
        return 1.0
    elif city in TIER1_CITIES:
        return 0.85 if willing_to_relocate else 0.55
    elif country == "India":
        return 0.5 if willing_to_relocate else 0.25
    else:
        return 0.3 if willing_to_relocate else 0.05


def behavioral_availability(signals):
    last_active = parse_date(signals.get("last_active_date"))
    days_inactive = (TODAY - last_active).days
    recency = float(np.exp(-max(0, days_inactive) / 90))
    return (
        0.35 * signals["recruiter_response_rate"]
        + 0.25 * signals["interview_completion_rate"]
        + 0.25 * recency
        + 0.15 * float(signals["open_to_work_flag"])
    )


# ── Main scoring function ───────────────────────────────────────────────────

def score_candidate(rec):
    """
    Returns (score: float, reasoning: str, flags: dict)
    score is in [0, 1]. Hard disqualifiers → score * 0.05.
    """
    profile  = rec["profile"]
    signals  = rec["redrob_signals"]
    skills   = rec["skills"]

    # Flags
    honeypot     = is_honeypot(rec)
    consulting   = is_consulting_only(rec)
    cv_speech    = is_cv_speech_only(rec)
    aspirational = is_aspirational_pivot(rec)
    any_disq     = honeypot or consulting or cv_speech

    # Scores
    exp_fit  = experience_fit(profile["years_of_experience"])
    sk_score = skill_tier_score(skills)
    tc_score = title_company_signal(
        profile["current_title"], profile["current_company"]
    )
    loc_score = location_fit(
        profile["location"],
        profile["country"],
        signals.get("willing_to_relocate", False)
    )
    beh_score = behavioral_availability(signals)

    # Composite
    base = (
        0.25 * exp_fit
        + 0.30 * sk_score
        + 0.30 * tc_score
        + 0.10 * beh_score
        + 0.05 * loc_score
    )

    # Penalties
    if any_disq:
        base *= 0.05
    elif aspirational:
        base *= 0.50

    base = round(float(base), 6)

    # Reasoning
    reasoning = _build_reasoning(profile, signals, skills)

    return base, reasoning, {
        "honeypot": honeypot,
        "consulting_only": consulting,
        "cv_speech_only": cv_speech,
        "aspirational": aspirational,
        "any_disq": any_disq,
    }


# ── Reasoning helpers ───────────────────────────────────────────────────────

def _pick_best_assessment(scores):
    if not scores:
        return None, None, False
    relevant = {k: v for k, v in scores.items() if k in NLP_IR_SKILLS}
    if relevant:
        skill = max(relevant, key=relevant.get)
        return skill, relevant[skill], True
    skill = max(scores, key=scores.get)
    return skill, scores[skill], False


def _build_caveats(notice_days, response_hours, response_rate):
    caveats = []
    if notice_days and notice_days > 60:
        caveats.append(f"{notice_days}-day notice period")
    if response_hours and response_hours > 96:
        caveats.append(f"slow response (~{response_hours/24:.0f}d)")
    if response_rate is not None and response_rate < 0.30:
        caveats.append(f"low response rate ({response_rate:.2f})")
    return caveats


def _build_reasoning(profile, signals, skills):
    num_ai = sum(1 for s in skills if s["name"] in NLP_IR_SKILLS)
    skill, assess_score, is_relevant = _pick_best_assessment(
        signals.get("skill_assessment_scores", {})
    )
    caveats = _build_caveats(
        signals.get("notice_period_days"),
        signals.get("avg_response_time_hours"),
        signals.get("recruiter_response_rate"),
    )
    if skill:
        a_clause = (
            f"strong {skill} assessment ({assess_score:.1f})" if is_relevant
            else f"best assessment {skill} ({assess_score:.1f}, not core-relevant)"
        )
    else:
        a_clause = "no assessment data"
    caveat_clause = f"; caveats: {', '.join(caveats)}" if caveats else ""
    return (
        f"{profile['current_title']} with {profile['years_of_experience']:.1f} yrs; "
        f"{num_ai} AI core skills; {a_clause}{caveat_clause}; {profile['location']}"
    )