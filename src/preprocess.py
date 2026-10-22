"""Prepare LawBot Q&A data for training and validation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

SEED = 42
SOURCE_BY_FILENAME = {
    "ipc_qa.json": "IPC",
    "crpc_qa.json": "CrPC",
    "constitution_qa.json": "Constitution",
}
SOURCES = ("IPC", "CrPC", "Constitution")
REFERENCE_RE = re.compile(
    r"\b(?:Sections?\s+\d+[A-Z]?(?:\s*[-–]\s*\d+[A-Z]?)?"
    r"|Articles?\s+\d+[A-Z]?(?:\s*\([\w]+\))?"
    r"|S\.\s*\d+[A-Z]?(?:\s*[-–]\s*\d+[A-Z]?)?(?:\s+(?:IPC|CrPC|BNS|BNSS))?)\b",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
MOJIBAKE = ("Ã", "Â", "â€™", "â€œ", "â€", "ï¿½")
GARBAGE_MARKERS = ("lorem ipsum", "[redacted]", "[deleted]", "null", "none", "n/a")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    # Repair the common UTF-8-as-Windows-1252 mojibake pattern when safe.
    if any(marker in text for marker in MOJIBAKE):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return SPACE_RE.sub(" ", text).strip()


def normalize_for_dedupe(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return SPACE_RE.sub(" ", re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)).strip()


def is_garbage(text: str, *, question: bool) -> bool:
    normalized = text.casefold().strip(" .?!")
    if not normalized or any(marker == normalized for marker in GARBAGE_MARKERS):
        return True
    if len(set(normalized)) <= 2:
        return True
    # Reject obvious cut-off fragments and non-prose symbol noise without
    # treating the datasets' ordinary short section-label answers as garbage.
    if text.endswith(("...", "…", "[truncated]", "<|endoftext|>")):
        return True
    alnum = sum(ch.isalnum() for ch in text)
    if len(text) >= 20 and alnum / max(len(text), 1) < 0.35:
        return True
    if question and len(text.split()) < 2:
        return True
    return False


def extract_references(*texts: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in REFERENCE_RE.finditer(text):
            reference = SPACE_RE.sub(" ", match.group(0)).strip(" .,:;")
            key = reference.casefold()
            if key not in seen:
                found.append(reference)
                seen.add(key)
    return found


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_for_dedupe(text)
    words = normalized.split()
    if len(words) < 2:
        return set(words)
    # Word bigrams substantially reduce sketch cost while retaining phrase order.
    return {f"{left} {right}" for left, right in zip(words, words[1:])}


def minhash_signature(grams: set[str], seeds: list[int]) -> tuple[int, ...]:
    if not grams:
        return tuple(0 for _ in seeds)
    # Hash each gram once, then use seeded affine permutations over a prime
    # field. A compact 16-permutation sketch keeps this practical at dataset scale.
    modulus = (1 << 61) - 1
    values = [int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big") % modulus
              for gram in grams]
    signature = []
    for seed in seeds:
        a = (seed * 2 + 1) % modulus
        b = (seed * 0x9E3779B97F4A7C15) % modulus
        signature.append(min((a * value + b) % modulus for value in values))
    return tuple(signature)


def remove_near_duplicates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Use deterministic seeded MinHash LSH and a >.95 signature threshold."""
    rng = random.Random(SEED)
    seeds = [rng.randrange(1, 2**31) for _ in range(16)]
    bands, rows_per_band = 8, 2
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    signatures: list[tuple[int, ...]] = []
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        grams = char_ngrams(row["instruction"])
        signature = minhash_signature(grams, seeds)
        candidate_indexes: set[int] = set()
        for band in range(bands):
            key = (band, signature[band * rows_per_band : (band + 1) * rows_per_band])
            candidate_indexes.update(buckets.get(key, []))
        duplicate = any(
            sum(a == b for a, b in zip(signature, signatures[index])) / len(seeds) > 0.95
            for index in candidate_indexes
        )
        if duplicate:
            removed += 1
            continue
        index = len(kept)
        kept.append(row)
        signatures.append(signature)
        for band in range(bands):
            key = (band, signature[band * rows_per_band : (band + 1) * rows_per_band])
            buckets[key].append(index)
    return kept, removed


def load_file(path: Path) -> tuple[list[dict[str, Any]], int]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = next((v for v in data.values() if isinstance(v, list)), [data])
    else:
        records = []
    normalized: list[dict[str, Any]] = []
    invalid_schema = 0
    source = SOURCE_BY_FILENAME.get(path.name.lower())
    for record in records:
        if not isinstance(record, dict):
            invalid_schema += 1
            continue
        question = clean_text(record.get("instruction", record.get("question", "")))
        answer = clean_text(record.get("output", record.get("answer", "")))
        row_source = clean_text(record.get("source", "")) or source
        if row_source not in SOURCES or not question or not answer:
            invalid_schema += 1
            continue
        normalized.append({"instruction": question, "output": answer, "source": row_source})
    return normalized, invalid_schema


def split_by_source(rows: list[dict[str, Any]]) -> None:
    rng = random.Random(SEED)
    for source in SOURCES:
        indexes = [i for i, row in enumerate(rows) if row["source"] == source]
        rng.shuffle(indexes)
        train_count = int(len(indexes) * 0.8)
        for rank, index in enumerate(indexes):
            rows[index]["split"] = "train" if rank < train_count else "validation"


def length_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    lengths = [len(row[field].split()) for row in rows]
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    return {
        "count": len(lengths),
        "mean": round(mean(lengths), 2),
        "median": median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(raw_dir: Path, project_dir: Path) -> dict[str, Any]:
    file_counts: dict[str, dict[str, int]] = {}
    raw_samples: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    input_count = invalid_schema = 0
    for filename in SOURCE_BY_FILENAME:
        path = raw_dir / filename
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        source_records = loaded if isinstance(loaded, list) else next(
            (v for v in loaded.values() if isinstance(v, list)), [loaded]
        ) if isinstance(loaded, dict) else []
        file_input = len(source_records)
        input_count += file_input
        normalized, bad = load_file(path)
        invalid_schema += bad
        file_counts[filename] = {"input": file_input, "valid_schema": len(normalized), "invalid_schema": bad}
        for raw_record in source_records:
            if isinstance(raw_record, dict) and len(raw_samples) < 3:
                raw_samples.append({"question": str(raw_record.get("question", raw_record.get("instruction", ""))),
                                    "answer": str(raw_record.get("answer", raw_record.get("output", "")))})
        rows.extend(normalized)

    schema_rows = len(rows)
    long_enough = [r for r in rows if len(r["instruction"]) >= 10 and len(r["output"]) >= 10]
    short_removed = schema_rows - len(long_enough)
    quality_rows = [r for r in long_enough if not is_garbage(r["instruction"], question=True)
                    and not is_garbage(r["output"], question=False)]
    garbage_removed = len(long_enough) - len(quality_rows)
    exact_seen: set[tuple[str, str]] = set()
    exact_rows: list[dict[str, Any]] = []
    exact_removed = 0
    for row in quality_rows:
        key = (normalize_for_dedupe(row["instruction"]), normalize_for_dedupe(row["output"]))
        if key in exact_seen:
            exact_removed += 1
        else:
            exact_seen.add(key)
            exact_rows.append(row)
    near_rows, near_removed = remove_near_duplicates(exact_rows)

    for row in near_rows:
        row["references"] = extract_references(row["instruction"], row["output"])
    split_by_source(near_rows)
    train = [r for r in near_rows if r["split"] == "train"]
    validation = [r for r in near_rows if r["split"] == "validation"]

    processed = project_dir / "data" / "processed"
    write_jsonl(processed / "lawbot_cleaned.jsonl", near_rows)
    write_jsonl(processed / "train.jsonl", train)
    write_jsonl(processed / "val.jsonl", validation)

    source_counts = Counter(row["source"] for row in near_rows)
    split_counts = Counter(row["split"] for row in near_rows)
    referenced = sum(bool(row["references"]) for row in near_rows)
    report = [
        "# LawBot preprocessing report", "", "## Input and row counts", "",
        "| File | Input rows | Valid schema | Invalid schema |", "|---|---:|---:|---:|",
    ]
    for name, counts in file_counts.items():
        report.append(f"| `{name}` | {counts['input']} | {counts['valid_schema']} | {counts['invalid_schema']} |")
    report.extend([
        f"| **Total** | **{input_count}** | **{schema_rows}** | **{invalid_schema}** |", "",
        "| Step | Rows remaining | Rows removed at step |", "|---|---:|---:|",
        f"| Valid schema normalization | {schema_rows} | {invalid_schema} invalid records |",
        f"| Minimum lengths (10 characters each) | {len(long_enough)} | {short_removed} |",
        f"| Garbage/truncation filter | {len(quality_rows)} | {garbage_removed} |",
        f"| Exact deduplication | {len(exact_rows)} | {exact_removed} |",
        f"| Near-duplicate question filter (MinHash > 0.95) | {len(near_rows)} | {near_removed} |",
        f"| Final train / validation | {len(train)} / {len(validation)} | — |", "",
        "## Source distribution", "", "| Source | Total | Train | Validation |", "|---|---:|---:|---:|",
    ])
    for source in SOURCES:
        srows = [r for r in near_rows if r["source"] == source]
        report.append(f"| {source} | {source_counts[source]} | {sum(r['split']=='train' for r in srows)} | {sum(r['split']=='validation' for r in srows)} |")
    report.extend(["", "## Length statistics (whitespace-separated words)", "", "| Field | Count | Mean | Median | Min | Max |", "|---|---:|---:|---:|---:|---:|"])
    for field, title in (("instruction", "Question"), ("output", "Answer")):
        stats = length_stats(near_rows, field)
        report.append(f"| {title} | {stats['count']} | {stats['mean']} | {stats['median']} | {stats['min']} | {stats['max']} |")
    report.extend(["", "## Extracted references", "", f"{referenced} of {len(near_rows)} records ({referenced / max(1, len(near_rows)):.1%}) contain at least one extracted section/article reference.", "",
                   "## Before/after examples", "", "Raw examples are shown alongside their cleaned, normalized records. The raw field mapping is `question`/`answer` to `instruction`/`output`.", ""])
    for i, raw in enumerate(raw_samples[:3]):
        cleaned = next((r for r in near_rows if normalize_for_dedupe(r["instruction"]) == normalize_for_dedupe(clean_text(raw["question"]))), None)
        report.extend([f"### Example {i + 1}", "", f"- Before question: {raw['question']}", f"- Before answer: {raw['answer']}"])
        if cleaned:
            report.extend([f"- After instruction: {cleaned['instruction']}", f"- After output: {cleaned['output']}", f"- References: {cleaned['references']}"])
        else:
            report.append("- After: filtered or superseded during deduplication.")
        report.append("")
    report.extend(["## Notes and limitations", "",
                   "- Inputs contain `question` and `answer`; the processor maps these to `instruction` and `output`. The four optional CrPC `id` fields are not used.",
                   "- Source labels are inferred from filenames because the input rows do not provide a `source` key.",
                   "- Near-duplicate matching uses 16 seeded word-bigram MinHash permutations and removes questions with estimated similarity strictly above 0.95 (all signature positions must agree at this sketch size).",
                   "- Reference extraction retains matches from both questions and answers. Original answer text remains intact after text cleanup.",
                   "- The source data uses legacy IPC/CrPC terminology. The IPC, CrPC, and Evidence Act were replaced by the BNS, BNSS, and Bharatiya Sakshya Adhiniyam, effective 1 July 2024; this dataset does not map or verify those current-law equivalents.",
                   "- Source answers are retained as supplied and were not independently checked for legal accuracy or currency.", ""])
    report_path = project_dir / "reports" / "preprocessing_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {"counts": file_counts, "input": input_count, "schema": schema_rows,
            "short_removed": short_removed, "garbage_removed": garbage_removed,
            "exact_removed": exact_removed, "near_removed": near_removed,
            "final": len(near_rows), "train": len(train), "validation": len(validation),
            "source_counts": dict(source_counts), "references": referenced}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = preprocess(args.raw_dir, args.project_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
