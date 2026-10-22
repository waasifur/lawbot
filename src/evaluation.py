"""Small, dependency-light metrics for LawBot answer generation."""

from __future__ import annotations

import re
from collections.abc import Sequence

_CITATION_RE = re.compile(
    r"(?:\b(?:Sections?|Articles?|S\.?|Art\.?)\s*|§\s*)(\d+[A-Z]?(?:\s*[-–]\s*\d+[A-Z]?)?)",
    re.IGNORECASE,
)


def citation_numbers(text: str) -> set[str]:
    """Return normalized section/article identifiers cited in text."""
    return {re.sub(r"\s*[-–]\s*", "-", match).upper() for match in _CITATION_RE.findall(text or "")}


def reference_numbers(references: Sequence[str]) -> set[str]:
    """Extract only numbered legal references from preprocessing metadata."""
    return set().union(*(citation_numbers(reference) for reference in references)) if references else set()


def citation_match(prediction: str, references: Sequence[str]) -> bool | None:
    """Exact citation-set match; return None for examples without reference IDs."""
    expected = reference_numbers(references)
    if not expected:
        return None
    return expected == citation_numbers(prediction)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L F1, implemented without a heavyweight dependency."""
    pred = (prediction or "").lower().split()
    ref = (reference or "").lower().split()
    if not pred or not ref:
        return 0.0
    # One-row dynamic programming is enough to compute the LCS length.
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for j, ref_token in enumerate(ref, start=1):
            current.append(previous[j - 1] + 1 if token == ref_token else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def citation_accuracy(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, int | float]:
    """Aggregate exact citation-set accuracy over eligible examples."""
    matches = [citation_match(pred, refs) for pred, refs in zip(predictions, references)]
    eligible = [match for match in matches if match is not None]
    correct = sum(bool(match) for match in eligible)
    return {"correct": correct, "eligible": len(eligible), "accuracy": correct / len(eligible) if eligible else 0.0}
