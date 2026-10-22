"""Small deterministic reward components for the LawBot GRPO experiment.

These are weak programmatic proxies, not legal judgment. A person should inspect
the generated answers and preference candidates before training or relying on them.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Sequence

ABSTAIN_RE = re.compile(r"(?:don't|do not) have enough evidence|cannot answer|unable to answer|not enough information", re.I)
NUMBER_RE = re.compile(r"(?:\b(?:sections?|articles?|s\.?|art\.?)\s*|§\s*)(\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?)", re.I)
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def completion_text(completion: Any) -> str:
    """Accept TRL standard strings or conversational completion messages."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        return " ".join(str(item.get("content", "")) for item in completion if isinstance(item, dict)).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion or "").strip()


def reference_ids(references: Any) -> set[str]:
    if not isinstance(references, (list, tuple)):
        return set()
    return {re.sub(r"\s*[-–]\s*", "-", match.group(1)).upper()
            for item in references for match in NUMBER_RE.finditer(str(item))}


def _column(values: Any, length: int, default: Any) -> list[Any]:
    if values is None:
        return [default] * length
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values] * length


def citation_match_reward(completions: Sequence[Any], references: Any = None,
                          is_answerable: Any = None, **kwargs: Any) -> list[float]:
    """Reward matching reference section/article IDs; don't require IDs if none exist."""
    refs = _column(references, len(completions), [])
    answerable = _column(is_answerable, len(completions), True)
    rewards = []
    for completion, gold, can_answer in zip(completions, refs, answerable):
        if not can_answer:
            rewards.append(0.0)
            continue
        expected = reference_ids(gold)
        found = {re.sub(r"\s*[-–]\s*", "-", m.group(1)).upper()
                 for m in NUMBER_RE.finditer(completion_text(completion))}
        if not expected:
            rewards.append(0.0)
        elif expected.intersection(found):
            rewards.append(1.0)
        elif found:
            rewards.append(-0.35)
        else:
            rewards.append(-0.15)
    return rewards


def answer_similarity_reward(completions: Sequence[Any], reference_answer: Any = None,
                             is_answerable: Any = None, **kwargs: Any) -> list[float]:
    """Cheap normalized string-similarity proxy scaled to [0, 1]."""
    refs = _column(reference_answer, len(completions), "")
    answerable = _column(is_answerable, len(completions), True)
    rewards = []
    for completion, reference, can_answer in zip(completions, refs, answerable):
        if not can_answer or not str(reference).strip():
            rewards.append(0.0)
            continue
        pred = " ".join(TOKEN_RE.findall(completion_text(completion).lower()))
        gold = " ".join(TOKEN_RE.findall(str(reference).lower()))
        rewards.append(SequenceMatcher(None, pred, gold, autojunk=False).ratio())
    return rewards


def length_format_reward(completions: Sequence[Any], is_answerable: Any = None,
                         **kwargs: Any) -> list[float]:
    """Mildly penalize empty, overly long, or citation-dense responses."""
    answerable = _column(is_answerable, len(completions), True)
    rewards = []
    for completion, can_answer in zip(completions, answerable):
        text = completion_text(completion)
        words = TOKEN_RE.findall(text)
        penalty = -0.5 if not words else 0.0
        if len(words) > 120:
            penalty -= min(0.5, (len(words) - 120) / 240)
        if len(NUMBER_RE.findall(text)) > 5:
            penalty -= 0.15
        if can_answer and ABSTAIN_RE.search(text):
            penalty -= 0.15
        rewards.append(float(max(-1.0, penalty)))
    return rewards


def abstention_reward(completions: Sequence[Any], is_answerable: Any = None,
                      **kwargs: Any) -> list[float]:
    """Reward abstention only for deliberately unanswerable prompts."""
    answerable = _column(is_answerable, len(completions), True)
    rewards = []
    for completion, can_answer in zip(completions, answerable):
        abstains = bool(ABSTAIN_RE.search(completion_text(completion)))
        rewards.append((-0.5 if abstains else 0.5) if not can_answer else (-0.2 if abstains else 0.0))
    return rewards
