"""Small auditable routing loop for RAG, local/API tools, and parametric answers."""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any, Callable

from src.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

ABSTAIN = "I don't have enough evidence in my sources to answer this"
LEGAL_CUES = re.compile(
    r"\b(?:law|legal|section|article|ipc|crpc|bns|bnss|constitution|court|bail|offence|offense|"
    r"murder|arrest|rights?|contract|crime|criminal|civil|judgment|precedent|act|statute|"
    r"limitation|filing|petition|prosecution|evidence|writ)\b", re.I)
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DAYS_PATTERN = re.compile(r"\b(\d{1,5})\s*(?:calendar\s+days?|business\s+days?|working\s+days?|days?)\b", re.I)


class LawBotAgent:
    """Route to case search, glossary, deadline calculator, RAG, or a model callback.

    The routing heuristic is deliberately visible and replaceable. An optional
    ``route_selector`` may return ``{"tool": name, "arguments": {...}}`` to wire
    a model-based JSON tool-calling response into the same audited execution path.
    """

    def __init__(self, rag_handler: Callable[..., Any] | None = None,
                 direct_answer: Callable[..., str] | None = None,
                 route_selector: Callable[..., dict[str, Any] | None] | None = None,
                 project_dir: str | None = None, tools_enabled: bool = True,
                 rag_enabled: bool = True):
        self.rag_handler = rag_handler
        self.direct_answer = direct_answer
        self.route_selector = route_selector
        self.project_dir = project_dir
        self.tools_enabled = tools_enabled
        self.rag_enabled = rag_enabled
        self.tool_schemas = TOOL_SCHEMAS

    def _route(self, message: str, history: list[dict[str, str]] | None = None) -> tuple[str, dict[str, Any]]:
        if self.tools_enabled and self.route_selector:
            selected = self.route_selector(message, history or [], self.tool_schemas)
            if selected:
                tool = selected.get("tool") or selected.get("name")
                if tool in TOOL_FUNCTIONS:
                    return "tool", {"tool": tool, "arguments": selected.get("arguments", {})}
        text = message.strip()
        lower = text.casefold()

        if self.tools_enabled:
            match = re.search(r"\bdefine\s+(?:the\s+)?(.+?)[?.!]*$", text, re.I)
            if not match:
                match = re.search(r"\bwhat\s+does\s+(.+?)\s+mean\??$", text, re.I)
            if not match:
                match = re.search(r"\bmeaning\s+of\s+(.+?)[?.!]*$", text, re.I)
            if match:
                term = match.group(1).strip(" \t?.!")
                # Use the tiny glossary only for terms it actually contains.
                # General vocabulary questions should go to the answer model.
                lookup = TOOL_FUNCTIONS["define_legal_term"](term, project_dir=self.project_dir)
                if lookup.get("definition"):
                    return "tool", {"tool": "define_legal_term", "arguments": {"term": term}}

        if self.tools_enabled and re.search(r"\b(case\s+law|judgment|judgement|precedent|find\s+(?:a\s+)?case|search\s+(?:for\s+)?cases?)\b", lower):
            query = re.sub(r"\b(?:find|search|case\s+law|judgments?|precedents?)\b", " ", text, flags=re.I)
            query = re.sub(r"\s+", " ", query).strip(" :-?.") or text
            return "tool", {"tool": "search_case_law", "arguments": {"query": query}}

        if self.tools_enabled and re.search(r"\b(deadline|due\s+date|calculate\s+(?:a\s+)?date|date\s+calculator)\b", lower):
            start = DATE_PATTERN.search(text)
            count = DAYS_PATTERN.search(text)
            if not start or not count:
                return "tool", {"tool": "calculate_deadline", "arguments": {}}
            business = bool(re.search(r"\b(?:business|working)\s+days?\b", count.group(0), re.I))
            exclude_weekends = bool(re.search(r"\bexcluding\s+weekends?\b|\bskip\s+weekends?\b", lower))
            args = {"start_date": start.group(1), "days": int(count.group(1)),
                    "kind": "business" if business else "calendar",
                    "exclude_weekends": True if business or exclude_weekends else False,
                    "exclude_holidays": bool(re.search(r"\bexcluding\s+holidays?\b", lower))}
            return "tool", {"tool": "calculate_deadline", "arguments": args}

        # When retrieval is enabled in settings but its handler/index is missing,
        # legal questions must still reach the configured answer model.
        if self.rag_enabled and self.rag_handler and LEGAL_CUES.search(text):
            return "rag", {}
        return "direct", {}

    def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a named JSON-schema tool and return one trace record."""
        if name not in TOOL_FUNCTIONS:
            raise ValueError(f"Unknown tool: {name}")
        args = dict(arguments or {})
        if name == "calculate_deadline" and not args:
            return {"error": "Provide start_date (YYYY-MM-DD) and days to calculate an illustrative date."}
        if name in {"search_case_law", "define_legal_term"}:
            args.setdefault("project_dir", self.project_dir)
        started = time.perf_counter()
        try:
            result = TOOL_FUNCTIONS[name](**args)
        except Exception as exc:
            result = {"mode": "tool_error", "error": f"{type(exc).__name__}: {exc}"}
        latency = round(time.perf_counter() - started, 6)
        return {"tool_name": name, "arguments": {k: v for k, v in args.items() if k != "token"},
                "result": result, "latency_seconds": latency}

    @staticmethod
    def _tool_answer(trace: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
        name, result = trace["tool_name"], trace["result"]
        if name == "define_legal_term":
            if result.get("definition"):
                answer = f"{result['term']}: {result['definition']}\n\n{result['notice']}"
            else:
                choices = result.get("suggestions") or result.get("available_terms", [])[:8]
                answer = (f"No exact glossary definition was found for ‘{result.get('term', '')}’. "
                          f"{result.get('notice', '')} Available terms: {', '.join(choices) or 'none listed'}.")
            return answer, ([{"source": result.get("source", "Local glossary"),
                              "term": result.get("term"), "url": result.get("source_url")}]
                            if result.get("definition") else []), "local_glossary"
        if name == "calculate_deadline":
            if result.get("deadline"):
                answer = (f"Illustrative calculated date: {result['deadline']} ({result['days']} {result['kind']} days; "
                          f"{result['counting_convention']}). {result['notice']} "
                          f"{result.get('holiday_calendar_note') or ''}")
            else:
                answer = result.get("error", "The date calculation could not be completed.")
            return answer, [], "illustrative_calculation"
        if name == "search_case_law":
            items = result.get("results", [])
            if not items:
                answer = "No matching result was found in the available search source. This does not establish that no relevant case exists. " + result.get("notice", "")
            else:
                blocks = []
                for item in items:
                    citation = f" ({item['citation']})" if item.get("citation") else ""
                    blocks.append(f"- [{item.get('title', 'Untitled case')}{citation}]({item.get('url', '')})\n  {item.get('snippet', '')}")
                answer = "Case-law search results (open and verify each original decision):\n" + "\n".join(blocks) + "\n\n" + result.get("attribution", "") + " " + result.get("notice", "")
            return answer, items, result.get("mode", "case_law_search")
        return json.dumps(result, ensure_ascii=False), [], "tool_result"

    def run(self, message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Route one user turn and return answer, grounding label, citations and latency trace."""
        history = history or []
        route, details = self._route(message, history)
        traces: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        if route == "tool":
            trace = self.run_tool(details["tool"], details.get("arguments", {}))
            traces.append(trace)
            answer, citations, grounding = self._tool_answer(trace)
        elif route == "rag":
            if not self.rag_handler:
                answer, grounding = ABSTAIN + ". The retrieval index or RAG handler is not available.", "no_retrieval_source"
            else:
                started = time.perf_counter()
                payload = self.rag_handler(message, history=history)
                latency = round(time.perf_counter() - started, 6)
                if isinstance(payload, str):
                    payload = {"answer": payload, "sources": []}
                citations = payload.get("sources", payload.get("citations", [])) or []
                if citations:
                    answer = payload.get("answer") or "Retrieved evidence is available; review the cited source excerpts."
                    grounding = "retrieved_sources"
                else:
                    answer = payload.get("answer") or ABSTAIN
                    if ABSTAIN not in answer:
                        answer = ABSTAIN + ". No retrieved source was provided."
                    grounding = "no_retrieval_source"
                traces.append({"tool_name": "rag_retrieval", "arguments": {"query": message},
                               "result": {"source_count": len(citations), "sources": citations},
                               "latency_seconds": latency})
        else:
            if not self.direct_answer:
                answer, grounding = ABSTAIN + ". No direct-answer model is configured.", "no_answer_model"
            else:
                payload = self.direct_answer(message, history=history)
                if isinstance(payload, dict):
                    answer = str(payload.get("answer", "")).strip()
                    grounding = str(payload.get("grounding", "parametric_unverified"))
                else:
                    answer, grounding = str(payload).strip(), "parametric_unverified"
                if grounding == "parametric_unverified" and LEGAL_CUES.search(message):
                    disclosure = "This response is from model knowledge and is unverified; no source was consulted."
                    if answer and disclosure.casefold() not in answer.casefold():
                        answer = f"{answer}\n\n{disclosure}"
        return {"answer": answer, "route": route, "grounding": grounding,
                "citations": citations, "tool_trace": traces}
