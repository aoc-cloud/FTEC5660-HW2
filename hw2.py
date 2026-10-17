#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# CV verification agent
#
# 1. The LLM reads each CV into a dictionary.
# 2. We search LinkedIn (and Facebook if needed) and pick the best profile.
# 3. We compare the CV with that profile using simple rules.
# 4. The LLM only judges differences that the rules cannot decide.

INDUSTRIES = [
    "AI", "Consulting", "Design", "Education", "Finance", "Healthcare",
    "Legal", "Logistics", "Marketing", "Software", "Engineering", "Sales",
    "Human Resources", "Operations", "Media", "Real Estate", "Hospitality",
    "Manufacturing", "Retail", "Telecommunications",
]

SCHOOL_ALIASES = {
    "hku": "university of hong kong",
    "hkust": "hong kong university of science and technology",
    "cuhk": "chinese university of hong kong",
    "polyu": "hong kong polytechnic university",
    "cityu": "city university of hong kong",
}

SKILL_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "powerpoint": "microsoft powerpoint",
    "excel": "microsoft excel",
    "word": "microsoft word",
    "ui ux design": "ui ux",
    "nodejs": "node js",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "powerbi": "power bi",
    "cicd": "ci cd",
}

FIELD_ALIASES = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "it": "information technology",
    "cs": "computer science",
}

DEGREES = {
    "phd": "phd", "doctor": "phd", "doctorate": "phd", "dphil": "phd",
    "master": "master", "mba": "master", "msc": "master", "meng": "master",
    "mfin": "master", "ma": "master", "mphil": "master", "mpa": "master",
    "med": "master", "mcom": "master",
    "bachelor": "bachelor", "bsc": "bachelor", "beng": "bachelor",
    "bfin": "bachelor", "ba": "bachelor", "bba": "bachelor",
    "bcom": "bachelor", "bed": "bachelor",
    "associate": "associate",
}

TITLE_PREFIXES = {"senior", "junior", "lead", "principal", "head", "chief", "staff", "associate"}
TITLE_LEVELS = {
    "junior": 1, "mid": 2, "senior": 3,
    "lead": 3, "principal": 3, "staff": 3, "head": 4, "chief": 4,
}

VALID_SCORE = 0.9
INVALID_SCORE = 0.1
MAX_PROFILES = 8


def norm(value):
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def company_key(value):
    words = norm(value).split()
    suffixes = {
        "ltd", "limited", "inc", "corp", "co", "company", "plc", "llc",
        "llp", "pte", "sdn", "bhd", "group", "holdings",
    }
    while words and words[-1] in suffixes:
        words.pop()
    return " ".join(words)


def school_key(value):
    key = norm(value)
    key = SCHOOL_ALIASES.get(key, key)
    return key[4:] if key.startswith("the ") else key


def skill_key(value):
    key = norm(value)
    if key.startswith("ms "):
        key = "microsoft " + key[3:]
    return SKILL_ALIASES.get(key, key)


def skill_match(left, right):
    left, right = skill_key(left), skill_key(right)
    if left == right:
        return True
    left, right = set(left.split()), set(right.split())
    return left <= right or right <= left


def field_key(value):
    key = norm(value)
    return FIELD_ALIASES.get(key, key)


def field_match(left, right):
    left, right = set(field_key(left).split()), set(field_key(right).split())
    return not left or not right or left <= right or right <= left


def clean_name(value):
    name = str(value or "").strip()
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    return name


def name_match(left, right):
    left, right = norm(clean_name(left)).split(), norm(clean_name(right)).split()
    if not left or not right:
        return True
    if left[-1] != right[-1]:
        return False
    return left[0] == right[0] or len(left[0]) == 1 or len(right[0]) == 1


def degree_level(value):
    for word in norm(value).split():
        if word in DEGREES:
            return DEGREES[word]
    return ""


def title_base(value):
    words = norm(value).split()
    while words and words[0] in TITLE_PREFIXES:
        words.pop(0)
    if words and words[0] == "of":
        words.pop(0)
    return " ".join(words)


def title_match(left, right):
    left, right = set(title_base(left).split()), set(title_base(right).split())
    return not left or not right or left <= right or right <= left


def title_level(value):
    words = norm(value).split()
    return words[0] if words and words[0] in TITLE_LEVELS else ""


def as_year(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def cv_years(cv):
    total = 0
    for job in cv.get("jobs", []):
        start = job.get("start_year")
        end = job.get("end_year")
        if end is None and job.get("is_current"):
            end = datetime.date.today().year
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            total += end - start
    return total


def text_of(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in value
        )
    if isinstance(value, dict):
        return json.dumps(value)
    content = getattr(value, "content", value)
    if content is not value:
        return text_of(content)
    return str(value)


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def clean_cv(data):
    jobs = []
    for job in data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        end_text = str(job.get("end_year") or "").lower()
        jobs.append({
            "company": str(job.get("company") or "").strip(),
            "title": str(job.get("title") or "").strip(),
            "start_year": as_year(job.get("start_year")),
            "end_year": as_year(job.get("end_year")),
            "is_current": bool(job.get("is_current")) or end_text in ("present", "current", "now"),
        })

    education = []
    for edu in data.get("education") or []:
        if not isinstance(edu, dict):
            continue
        education.append({
            "school": str(edu.get("school") or "").strip(),
            "degree": str(edu.get("degree") or "").strip(),
            "field": str(edu.get("field") or "").strip(),
            "graduation_year": as_year(edu.get("graduation_year")),
        })

    skills = []
    for skill in data.get("skills") or []:
        if isinstance(skill, dict):
            skill = skill.get("name")
        if skill:
            skills.append(str(skill).strip())

    city = str(data.get("city") or "").strip().split(",")[0].strip()

    return {
        "name": str(data.get("name") or "").strip(),
        "city": city,
        "headline": str(data.get("headline") or "").strip(),
        "jobs": jobs,
        "education": education,
        "skills": skills,
    }


def parse_ok(cv, text):
    text = norm(text)
    if not cv["name"] or norm(cv["name"]) not in text:
        return False
    if not cv["jobs"] and not cv["education"]:
        return False
    for job in cv["jobs"]:
        if job["company"] and company_key(job["company"]) not in text:
            return False
    for edu in cv["education"]:
        if edu["school"] and norm(edu["school"]) not in text:
            return False
    return True


def profile_score(cv, profile):
    experience = profile.get("experience") or []
    education = profile.get("education") or []
    skills = profile.get("skills") or []

    score = 0

    cv_companies = {company_key(job.get("company")) for job in cv.get("jobs", []) if job.get("company")}
    profile_companies = {company_key(job.get("company")) for job in experience if job.get("company")}
    score += 3 * len(cv_companies & profile_companies)

    cv_schools = {school_key(edu.get("school")) for edu in cv.get("education", []) if edu.get("school")}
    profile_schools = {school_key(edu.get("school")) for edu in education if edu.get("school")}
    score += 2 * len(cv_schools & profile_schools)

    cv_fields = {field_key(edu.get("field")) for edu in cv.get("education", []) if edu.get("field")}
    profile_fields = {field_key(edu.get("field")) for edu in education if edu.get("field")}
    score += len(cv_fields & profile_fields)

    cv_grad = {edu.get("graduation_year") for edu in cv.get("education", []) if edu.get("graduation_year")}
    profile_grad = {edu.get("end_year") for edu in education if edu.get("end_year")}
    score += len(cv_grad & profile_grad)

    profile_skills = [
        skill.get("name", "") if isinstance(skill, dict) else str(skill)
        for skill in skills
    ]
    cv_skills = [skill for skill in cv.get("skills", []) if skill]
    if cv_skills:
        matches = sum(1 for skill in cv_skills if any(skill_match(skill, other) for other in profile_skills))
        score += matches / len(cv_skills)

    if norm(cv.get("city")) and norm(cv.get("city")) == norm(profile.get("city")):
        score += 0.5

    years = profile.get("years_experience")
    if isinstance(years, (int, float)) and abs(years - cv_years(cv)) <= 2:
        score += 0.5

    return score


def compare(cv, profile):
    experience = profile.get("experience") or []
    education = profile.get("education") or []
    skills = profile.get("skills") or []
    diffs = []

    def add(reason, ask_llm=False):
        diffs.append({"reason": reason, "ask_llm": ask_llm, "genuine": None if ask_llm else True})

    if not name_match(cv.get("name"), profile.get("name")):
        add(f"Name: CV says {cv.get('name')}, profile says {profile.get('name')}")

    city = norm(cv.get("city"))
    profile_city = norm(profile.get("city"))
    profile_country = norm(profile.get("country"))
    if city and profile_city:
        a, b = set(city.split()), set(profile_city.split())
        if not (city == profile_city or city == profile_country or a <= b or b <= a):
            add(f"City: CV says {cv.get('city')}, profile says {profile.get('city')}")

    for job in cv.get("jobs", []):
        company = company_key(job.get("company"))
        matches = [entry for entry in experience if company_key(entry.get("company")) == company]
        if not matches:
            add(f"Employer {job.get('company')} is not in the profile")
            continue

        match = max(
            matches,
            key=lambda entry: (
                title_match(job.get("title"), entry.get("title")),
                -abs((job.get("start_year") or 0) - (entry.get("start_year") or 0)),
            ),
        )

        if not title_match(job.get("title"), match.get("title")):
            add(f"Title: CV says {job.get('title')}, profile says {match.get('title')}", ask_llm=True)

        level = title_level(job.get("title"))
        profile_level = norm(match.get("seniority"))
        if level and level not in norm(match.get("title")).split() and profile_level in ("junior", "mid", "senior"):
            if TITLE_LEVELS.get(level, 0) > TITLE_LEVELS.get(profile_level, 0):
                add(f"Seniority: CV says {job.get('title')}, profile seniority is {profile_level}")

        start = job.get("start_year")
        profile_start = match.get("start_year")
        if isinstance(start, int) and isinstance(profile_start, int) and start != profile_start:
            add(f"{job.get('company')} start year: CV says {start}, profile says {profile_start}")

        current = bool(job.get("is_current"))
        profile_current = bool(match.get("is_current")) or match.get("end_year") is None
        if current != profile_current:
            add(
                f"{job.get('company')}: CV says {'Present' if current else job.get('end_year')}, "
                f"profile says {'Present' if profile_current else match.get('end_year')}"
            )
        else:
            end = job.get("end_year")
            profile_end = match.get("end_year")
            if isinstance(end, int) and isinstance(profile_end, int) and end != profile_end:
                add(f"{job.get('company')} end year: CV says {end}, profile says {profile_end}")

    for edu in cv.get("education", []):
        school = school_key(edu.get("school"))
        matches = [entry for entry in education if school_key(entry.get("school")) == school]
        if not matches:
            add(f"School {edu.get('school')} is not in the profile")
            continue

        match = matches[0]
        for entry in matches:
            if degree_level(edu.get("degree")) == degree_level(entry.get("degree")):
                match = entry
                break

        cv_level = degree_level(edu.get("degree"))
        profile_level = degree_level(match.get("degree"))
        if cv_level and profile_level:
            if cv_level != profile_level:
                add(f"Degree: CV says {edu.get('degree')}, profile says {match.get('degree')}")
        elif norm(edu.get("degree")) != norm(match.get("degree")):
            add(f"Degree: CV says {edu.get('degree')}, profile says {match.get('degree')}", ask_llm=True)

        if edu.get("field") and match.get("field") and not field_match(edu.get("field"), match.get("field")):
            add(f"Field: CV says {edu.get('field')}, profile says {match.get('field')}", ask_llm=True)

        graduation = edu.get("graduation_year")
        profile_end = match.get("end_year")
        if isinstance(graduation, int) and isinstance(profile_end, int) and graduation != profile_end:
            add(f"Graduation year: CV says {graduation}, profile says {profile_end}")

    profile_skills = [
        skill.get("name", "") if isinstance(skill, dict) else str(skill)
        for skill in skills
    ]
    for skill in cv.get("skills", []):
        if skill and not any(skill_match(skill, other) for other in profile_skills):
            add(f"Skill {skill} is not in the profile", ask_llm=True)

    return diffs


class CVVerifierAgent:
    def __init__(self, tools):
        from langchain_deepseek import ChatDeepSeek

        self.llm = ChatDeepSeek(model=MODEL_NAME, temperature=0)
        self.tools = {tool.name: tool for tool in tools}

    async def call_tool(self, name, **args):
        args = {key: value for key, value in args.items() if value is not None}
        for attempt in range(3):
            try:
                result = await self.tools[name].ainvoke(args)
                text = text_of(result)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1 + attempt)

    async def ask_json(self, system, user):
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(2):
            try:
                response = await self.llm.ainvoke(messages)
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
                raise
            try:
                return extract_json(text_of(response.content))
            except Exception:
                if attempt == 0:
                    messages.append(HumanMessage(content="Return valid JSON only, without markdown."))
        raise ValueError("The model did not return valid JSON.")

    async def parse_cv(self, text):
        system = """You read a CV and return one JSON object.

{
  "name": "full name",
  "city": "city from the contact line",
  "headline": "the line under the name",
  "jobs": [{"company": "...", "title": "...", "start_year": 2020, "end_year": 2022, "is_current": false}],
  "education": [{"school": "...", "degree": "...", "field": "...", "graduation_year": 2020}],
  "skills": ["..."]
}

Copy values as written. Use null for a missing year. "Present" means is_current is true.
Ignore the job-description bullets."""

        for _ in range(2):
            data = await self.ask_json(system, f"CV TEXT:\n{text}")
            if isinstance(data, dict):
                cv = clean_cv(data)
                if parse_ok(cv, text):
                    return cv
        raise ValueError("Could not parse the CV.")

    async def linkedin_candidates(self, cv, industry):
        name = clean_name(cv.get("name"))
        searches = []
        if industry:
            searches.append({"q": name, "location": cv.get("city"), "industry": industry, "limit": 20})
        searches.append({"q": name, "location": cv.get("city"), "limit": 20})
        searches.append({"q": name, "limit": 20})

        for args in searches:
            try:
                rows = await self.call_tool("search_linkedin_people", **args)
            except Exception:
                continue
            if isinstance(rows, list) and rows:
                exact = [row for row in rows if isinstance(row, dict) and row.get("match_type") == "exact"]
                return exact or rows
        return []

    async def profiles_for(self, candidates, cv):
        candidates = [row for row in candidates if isinstance(row, dict)]
        years = cv_years(cv)
        candidates = sorted(candidates, key=lambda row: abs((row.get("years_experience") or 0) - years))

        profiles = []
        for row in candidates[:MAX_PROFILES]:
            if row.get("id") is None:
                continue
            try:
                profile = await self.call_tool("get_linkedin_profile", person_id=row["id"])
            except Exception:
                continue
            if isinstance(profile, dict) and not profile.get("error"):
                profiles.append(profile)
        return profiles

    async def facebook_fallback(self, cv):
        name = clean_name(cv.get("name"))
        try:
            users = await self.call_tool(
                "search_facebook_users", q=name, location=cv.get("city"), limit=5, fuzzy=True
            )
            if not users:
                return []
            fb = await self.call_tool("get_facebook_profile", user_id=users[0]["id"])
            legal_name = fb.get("original_name") or name
            rows = await self.call_tool(
                "search_linkedin_people", q=legal_name, location=cv.get("city"), limit=20
            )
        except Exception:
            return []

        profiles = []
        for row in (rows if isinstance(rows, list) else [])[:8]:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            try:
                profile = await self.call_tool("get_linkedin_profile", person_id=row["id"])
            except Exception:
                continue
            if isinstance(profile, dict) and not profile.get("error"):
                profiles.append(profile)
        return profiles

    async def find_profile(self, cv):
        industry = None
        headline = norm(cv.get("headline"))
        for name in INDUSTRIES:
            if norm(name) in headline:
                industry = name
                break

        if industry is None:
            titles = norm(" ".join(job.get("title", "") for job in cv.get("jobs", [])))
            for name in INDUSTRIES:
                if norm(name) in titles:
                    industry = name
                    break

        candidates = await self.linkedin_candidates(cv, industry)
        profiles = await self.profiles_for(candidates, cv)
        if not profiles:
            profiles = await self.facebook_fallback(cv)
        if not profiles:
            return None, 0

        scored = [(profile_score(cv, profile), profile) for profile in profiles]
        score, best = max(scored, key=lambda item: item[0])
        return best, score

    async def judge(self, diffs, cv, profile):
        questions = [diff for diff in diffs if diff["ask_llm"]]
        if not questions:
            return diffs

        system = """You are checking a CV against a LinkedIn profile.
For each difference, decide if it is only a wording difference or a real discrepancy.

Wording differences: BSc vs Bachelor of Science, UI/UX Design vs UI/UX,
a senior title when the profile seniority is senior, and a CV that lists fewer skills.

Real discrepancies: a higher job title, different years, a higher degree, a school
or employer that is not in the profile, a wrong city, or a skill the person does not have.

Return JSON only:
{"real": [indices], "wording": [indices]}"""

        payload = {
            "cv": cv,
            "profile": profile,
            "differences": [
                {"index": i, "reason": diff["reason"]}
                for i, diff in enumerate(questions)
            ],
        }
        try:
            answer = await self.ask_json(system, json.dumps(payload, ensure_ascii=False))
            real = {int(i) for i in answer.get("real", [])}
            wording = {int(i) for i in answer.get("wording", [])}
        except Exception:
            real, wording = set(range(len(questions))), set()

        for i, diff in enumerate(questions):
            diff["genuine"] = i in real or i not in wording
        return diffs

    async def verify(self, text):
        cv = await self.parse_cv(text)
        profile, confidence = await self.find_profile(cv)
        if profile is None:
            return INVALID_SCORE
        if confidence < 1:
            # The identity match is not strong enough to call the CV valid.
            return 0.5

        diffs = compare(cv, profile)
        diffs = await self.judge(diffs, cv, profile)
        return INVALID_SCORE if any(diff["genuine"] for diff in diffs) else VALID_SCORE


def build_agent(tools: list[Any]) -> Any:
    """Create the CV verifier agent."""
    return CVVerifierAgent(tools)


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Verify every CV and return {file_name: score}."""
    semaphore = asyncio.Semaphore(3)

    async def verify_one(name, text):
        async with semaphore:
            try:
                score = await asyncio.wait_for(agent.verify(text), timeout=150)
                if isinstance(score, (int, float)) and 0 <= score <= 1:
                    return name, float(score)
            except Exception:
                pass
            return name, 0.5

    results = await asyncio.gather(*(verify_one(name, text) for name, text in cvs.items()))
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
