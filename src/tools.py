"""Three small LawBot tools with JSON schemas and explicit source/fallback labels."""

from __future__ import annotations

import html
import json
import os
import re
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class _TextOnly(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _plain_text(value: str) -> str:
    parser = _TextOnly()
    parser.feed(html.unescape(value or ""))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _project_root(project_dir: str | Path | None = None) -> Path:
    if project_dir:
        return Path(project_dir).expanduser().resolve()
    if os.getenv("LAWBOT_PROJECT_DIR"):
        return Path(os.environ["LAWBOT_PROJECT_DIR"]).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _cached_case_search(query: str, project_dir: str | Path | None, mode: str,
                        error: str | None = None) -> dict[str, Any]:
    """Search already bundled official-summary paraphrases; never call these live results."""
    root = _project_root(project_dir)
    cache = root / "data" / "rag_corpus" / "supreme_court_summaries.jsonl"
    terms = {t for t in re.findall(r"[a-z0-9]{3,}", query.casefold())
             if t not in {"the", "and", "for", "from", "with", "case", "law", "about"}}
    hits: list[tuple[int, dict[str, Any]]] = []
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            content = " ".join(str(row.get(k, "")) for k in ("doc_title", "case_citation", "summary")).casefold()
            overlap = sum(term in content for term in terms)
            if overlap:
                hits.append((overlap, row))
    hits.sort(key=lambda item: (-item[0], item[1].get("case_citation", "")))
    results = [{
        "title": row.get("doc_title", "Supreme Court case summary"),
        "citation": row.get("case_citation", ""),
        "snippet": row.get("summary", ""),
        "url": row.get("source_url", "https://www.sci.gov.in/landmark-judgment-summaries/"),
        "source_type": "cached_official_summary_paraphrase",
        "notice": row.get("summary_notice", "Summary only; verify the full judgment."),
    } for _, row in hits[:5]]
    result: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "results": results,
        "notice": ("No Indian Kanoon API token was configured; this is a local cached-summary lookup, "
                   "not a live case-law search. Cached summaries are not judgments."),
        "attribution": "Cached paraphrases from the Supreme Court of India Landmark Judgment Summaries page.",
    }
    if error:
        result["api_error"] = error
    return result


def search_case_law(query: str, project_dir: str | Path | None = None,
                    token: str | None = None, timeout: int = 20) -> dict[str, Any]:
    """Search Indian Kanoon when INDIANKANOON_TOKEN is set; otherwise search local cached summaries."""
    query = str(query or "").strip()
    if not query:
        return {"query": query, "mode": "validation_error", "results": [], "error": "query is required"}
    if len(query) > 500:
        return {"query": query[:500], "mode": "validation_error", "results": [], "error": "query must be 500 characters or fewer"}
    api_token = token if token is not None else os.getenv("INDIANKANOON_TOKEN", "").strip()
    if not api_token:
        return _cached_case_search(query, project_dir, mode="cached_fallback")

    try:
        import requests

        response = requests.post(
            "https://api.indiankanoon.org/search/",
            params={"formInput": query, "pagenum": 0},
            headers={"Authorization": f"Token {api_token}", "Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        docs = payload.get("docs", []) if isinstance(payload, dict) else []
        results = []
        for row in docs[:5]:
            doc_id = row.get("tid")
            results.append({
                "title": _plain_text(str(row.get("title", ""))),
                "citation": _plain_text(str(row.get("docsource", ""))),
                "snippet": _plain_text(str(row.get("headline", ""))),
                "url": f"https://indiankanoon.org/doc/{doc_id}/" if doc_id else "https://indiankanoon.org/",
                "source_type": "indian_kanoon_api_search_result",
            })
        return {
            "query": query,
            "mode": "indian_kanoon_api",
            "found": payload.get("found"),
            "results": results,
            "attribution": "Search results provided by Indian Kanoon API. Open and verify the original judgment before relying on it.",
            "source_url": "https://api.indiankanoon.org/documentation/",
        }
    except Exception as exc:
        return _cached_case_search(query, project_dir, mode="cached_fallback_after_api_error",
                                   error=f"{type(exc).__name__}: live API search failed; token value omitted")


def define_legal_term(term: str, project_dir: str | Path | None = None) -> dict[str, Any]:
    """Look up an educational plain-language term in the bundled local glossary."""
    term = str(term or "").strip()
    if not term:
        return {"term": term, "mode": "validation_error", "error": "term is required"}
    path = _project_root(project_dir) / "data" / "legal_glossary.json"
    glossary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"terms": {}}
    entries = glossary.get("terms", {})
    normalized = {key.casefold(): (key, value) for key, value in entries.items()}
    key = term.casefold()
    if key in normalized:
        display, entry = normalized[key]
        return {"term": display, "definition": entry["definition"], "mode": "local_glossary",
                "source": entry.get("source", glossary.get("notice", "Local educational glossary")),
                "source_url": entry.get("source_url"),
                "notice": glossary.get("notice", "This general explanation is not a statutory definition.")}
    from difflib import get_close_matches

    suggestions = get_close_matches(key, normalized.keys(), n=3, cutoff=0.70)
    return {"term": term, "mode": "local_glossary", "definition": None,
            "suggestions": [normalized[item][0] for item in suggestions],
            "available_terms": list(entries),
            "notice": "No exact entry was found; suggestions are not definitions. Check authoritative statutes and judgments."}


def calculate_deadline(start_date: str, days: int, kind: str = "calendar",
                       exclude_weekends: bool | None = None, exclude_holidays: bool = False,
                       holidays: list[str] | None = None, include_start_date: bool = False) -> dict[str, Any]:
    """Calculate an illustrative date count, not a jurisdiction-specific legal limitation period."""
    try:
        start = date.fromisoformat(str(start_date))
    except (TypeError, ValueError):
        return {"mode": "validation_error", "error": "start_date must be YYYY-MM-DD"}
    if isinstance(days, bool) or not isinstance(days, int) or days < 0 or days > 36500:
        return {"mode": "validation_error", "error": "days must be an integer from 0 to 36500"}
    kind_normalized = str(kind or "calendar").casefold()
    if kind_normalized not in {"calendar", "business"}:
        return {"mode": "validation_error", "error": "kind must be 'calendar' or 'business'"}
    skip_weekends = (kind_normalized == "business") if exclude_weekends is None else bool(exclude_weekends)
    holiday_dates: set[date] = set()
    try:
        holiday_dates = {date.fromisoformat(value) for value in (holidays or [])}
    except (TypeError, ValueError):
        return {"mode": "validation_error", "error": "holidays must contain YYYY-MM-DD dates"}
    if include_start_date:
        current = start
        counted = 1 if (not skip_weekends or current.weekday() < 5) and (not exclude_holidays or current not in holiday_dates) else 0
    else:
        current = start
        counted = 0
    excluded_dates: list[str] = []
    while counted < days:
        current += timedelta(days=1)
        weekend = current.weekday() >= 5
        holiday = current in holiday_dates
        if (skip_weekends and weekend) or (exclude_holidays and holiday):
            excluded_dates.append(current.isoformat())
            continue
        counted += 1
    convention = "start date excluded" if not include_start_date else "start date counted when eligible"
    return {
        "mode": "illustrative_date_calculation", "start_date": start.isoformat(),
        "deadline": current.isoformat(), "days": days, "kind": kind_normalized,
        "exclude_weekends": skip_weekends, "exclude_holidays": bool(exclude_holidays),
        "holidays_considered": sorted(d.isoformat() for d in holiday_dates) if exclude_holidays else [],
        "excluded_dates": excluded_dates, "counting_convention": convention,
        "notice": "Illustrative date arithmetic only. Legal limitation and filing rules may count days differently and may have court-specific holidays or exceptions.",
        "holiday_calendar_note": ("No holiday dates were supplied, so no public holidays were excluded."
                                  if exclude_holidays and not holiday_dates else None),
    }


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_case_law", "description": "Search Indian Kanoon case law; when no API token is configured, use and label a local cached-summary fallback.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Non-confidential legal search query."}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "define_legal_term", "description": "Look up an educational plain-language definition in LawBot's local legal glossary.", "parameters": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculate_deadline", "description": "Calculate an illustrative date offset; this does not determine a legal limitation deadline.", "parameters": {"type": "object", "properties": {"start_date": {"type": "string", "format": "date"}, "days": {"type": "integer", "minimum": 0}, "kind": {"type": "string", "enum": ["calendar", "business"]}, "exclude_weekends": {"type": ["boolean", "null"]}, "exclude_holidays": {"type": "boolean"}, "holidays": {"type": "array", "items": {"type": "string", "format": "date"}}, "include_start_date": {"type": "boolean"}}, "required": ["start_date", "days"], "additionalProperties": False}}},
]


TOOL_FUNCTIONS = {"search_case_law": search_case_law,
                  "define_legal_term": define_legal_term,
                  "calculate_deadline": calculate_deadline}
