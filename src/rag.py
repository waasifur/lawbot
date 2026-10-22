"""Download, clean, chunk, index, and retrieve LawBot RAG sources."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SECTIONS = re.compile(
    r"^\s*(?:(?:section|article|art\.?|s\.)\s*)?(\d{1,3}[A-Z]?)\s*[.)]\s+([^\n]{4,180})\s*$",
    re.IGNORECASE,
)
EXPLICIT_SECTION = re.compile(r"^\s*(?:section|article|art\.?|s\.)\s+(\d{1,3}[A-Z]?)\b", re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,4}(?:\s*[-–—])?\s*$")
WHITESPACE = re.compile(r"[\t \u00a0]+")
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=3600,
    chunk_overlap=480,
    length_function=len,
    separators=["\n\n", "\n", ". ", "; ", " ", ""],
)


def read_manifest(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "data" / "rag_corpus" / "source_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def download_official_pdfs(project_dir: Path, timeout: int = 60) -> list[Path]:
    """Fetch only PDFs whose host is the India Code domain in the project manifest."""
    import requests

    manifest = read_manifest(project_dir)
    raw_dir = project_dir / "data" / "rag_corpus" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for item in manifest["pdf_documents"]:
        url = item["source_url"]
        host = (urlparse(url).hostname or "").lower()
        if not (host == "indiacode.nic.in" or host.endswith(".indiacode.nic.in")):
            raise ValueError(f"Refusing non-India-Code download URL: {url}")
        destination = raw_dir / item["filename"]
        if not destination.exists():
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "LawBot student research corpus"})
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"Expected a PDF but received non-PDF content from {url}")
            destination.write_bytes(response.content)
        downloaded.append(destination)
        print(f"Ready: {item['title']} -> {destination} ({destination.stat().st_size:,} bytes)")
    return downloaded


def pdf_pages(pdf_path: Path) -> list[str]:
    """Return one extracted text string per PDF page using PyMuPDF or pdfplumber."""
    try:
        import fitz

        with fitz.open(pdf_path) as pdf:
            return [page.get_text("text", sort=True) for page in pdf]
    except ImportError:
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError("Install pymupdf or pdfplumber to extract PDFs.") from exc
        with pdfplumber.open(pdf_path) as pdf:
            return [page.extract_text(x_tolerance=2, y_tolerance=3) or "" for page in pdf.pages]


def clean_repeated_page_furniture(pages: list[str]) -> list[str]:
    """Remove recurring page headers/footers and standalone page numbers."""
    normalized = [[WHITESPACE.sub(" ", line).strip() for line in page.splitlines()] for page in pages]
    edge_lines = []
    for lines in normalized:
        nonempty = [line for line in lines if line]
        edge_lines.extend(nonempty[:2])
        edge_lines.extend(nonempty[-2:])
    counts = Counter(line.casefold() for line in edge_lines if len(line) > 3)
    repeat_threshold = max(3, int(len(pages) * 0.15))
    recurring = {line for line, count in counts.items() if count >= repeat_threshold}
    cleaned = []
    for lines in normalized:
        filtered = [line for line in lines if not PAGE_NUMBER.match(line) and line.casefold() not in recurring]
        text = "\n".join(filtered)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned.append(text)
    return cleaned


def heading_section(line: str, act: str) -> str | None:
    explicit = EXPLICIT_SECTION.match(line)
    if explicit:
        return explicit.group(1).upper()
    match = SECTIONS.match(line)
    if match:
        return match.group(1).upper()
    return None


def section_units(pages: list[str], act: str) -> list[dict[str, Any]]:
    """Split text at statute section or Constitution article boundaries."""
    units: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_section: str | None = None
    start_page = 1
    end_page = 1

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if len(text) >= 25:
            units.append({"text": text, "section_number": current_section,
                          "page_start": start_page, "page_end": end_page})

    for page_num, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            section_number = heading_section(line, act)
            if section_number is not None and current_lines:
                flush()
                current_lines = []
                current_section = section_number
                start_page = page_num
            elif section_number is not None and not current_lines:
                current_section = section_number
                start_page = page_num
            current_lines.append(line)
            end_page = page_num
    flush()
    return units


def group_section_units(units: list[dict[str, Any]], target_chars: int = 3200) -> list[dict[str, Any]]:
    """Join adjacent short sections; large sections remain independently chunkable."""
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for unit in units:
        if current and len(current["text"]) + len(unit["text"]) + 2 > target_chars:
            groups.append(current)
            current = None
        if current is None:
            current = {"text": unit["text"], "section_numbers": [],
                       "page_start": unit["page_start"], "page_end": unit["page_end"]}
        else:
            current["text"] += "\n\n" + unit["text"]
            current["page_end"] = unit["page_end"]
        if unit["section_number"] and unit["section_number"] not in current["section_numbers"]:
            current["section_numbers"].append(unit["section_number"])
    if current:
        groups.append(current)
    return groups


def document_chunks(pdf_path: Path, doc_meta: dict[str, Any]) -> list[Document]:
    pages = clean_repeated_page_furniture(pdf_pages(pdf_path))
    units = section_units(pages, doc_meta["act"])
    groups = group_section_units(units)
    docs: list[Document] = []
    for group in groups:
        parts = SPLITTER.split_text(group["text"])
        for part in parts:
            if len(part.strip()) < 25:
                continue
            section_numbers = group["section_numbers"]
            docs.append(Document(
                page_content=part,
                metadata={
                    "doc": doc_meta["filename"],
                    "doc_title": doc_meta["title"],
                    "act": doc_meta["act"],
                    "section_number": section_numbers[0] if len(section_numbers) == 1 else None,
                    "section_numbers": section_numbers,
                    "page": group["page_start"] if group["page_start"] == group["page_end"] else f"{group['page_start']}-{group['page_end']}",
                    "page_start": group["page_start"],
                    "page_end": group["page_end"],
                    "source_url": doc_meta["source_url"],
                    "content_type": "bare_act_pdf",
                },
            ))
    return docs


def summary_documents(project_dir: Path) -> list[Document]:
    path = project_dir / "data" / "rag_corpus" / "supreme_court_summaries.jsonl"
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = f"{row['doc_title']}\n{row['case_citation']}\n{row['summary']}"
        docs.append(Document(page_content=text, metadata={
            "doc": "supreme_court_summaries.jsonl",
            "doc_title": row["doc_title"],
            "case_citation": row["case_citation"],
            "act": "Supreme Court summary",
            "section_number": None,
            "section_numbers": [],
            "page": None,
            "page_start": None,
            "page_end": None,
            "source_url": row["source_url"],
            "summary_notice": row["summary_notice"],
            "content_type": "official_summary_paraphrase",
        }))
    return docs


def build_corpus_chunks(project_dir: Path, download: bool = True) -> list[Document]:
    manifest = read_manifest(project_dir)
    pdf_paths = download_official_pdfs(project_dir) if download else [
        project_dir / "data" / "rag_corpus" / "raw" / item["filename"]
        for item in manifest["pdf_documents"]
    ]
    all_docs: list[Document] = []
    for item, pdf_path in zip(manifest["pdf_documents"], pdf_paths):
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing source PDF {pdf_path}; run download_official_pdfs first.")
        page_text = clean_repeated_page_furniture(pdf_pages(pdf_path))
        text_path = project_dir / "data" / "rag_corpus" / "extracted_text" / f"{item['act'].lower()}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n\n".join(f"[Page {i}]\n{page}" for i, page in enumerate(page_text, start=1)), encoding="utf-8")
        docs = document_chunks(pdf_path, item)
        print(f"{item['act']}: {len(page_text)} pages -> {len(docs)} chunks")
        all_docs.extend(docs)
    all_docs.extend(summary_documents(project_dir))
    for index, doc in enumerate(all_docs):
        doc.metadata["chunk_id"] = f"chunk-{index:06d}"

    output = project_dir / "data" / "rag_corpus" / "corpus_chunks.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in all_docs:
            handle.write(json.dumps({"text": doc.page_content, **doc.metadata}, ensure_ascii=False) + "\n")
    print_chunk_statistics(all_docs)
    return all_docs


def print_chunk_statistics(docs: list[Document]) -> dict[str, Any]:
    counts = Counter(doc.metadata["act"] for doc in docs)
    tokens = [max(1, round(len(doc.page_content) / 4)) for doc in docs]
    stats = {"chunks": len(docs), "by_act": dict(counts),
             "tokens_estimate": {"mean": round(sum(tokens) / max(len(tokens), 1), 1),
                                 "min": min(tokens, default=0), "max": max(tokens, default=0)},
             "under_600_tokens": sum(n < 600 for n in tokens), "over_1000_tokens": sum(n > 1000 for n in tokens)}
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def make_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {"device": device} if device else {}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(docs: list[Document], project_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_community.vectorstores import FAISS

    embeddings = make_embeddings(model_name)
    vectorstore = FAISS.from_documents(docs, embeddings, normalize_L2=True)
    index_dir = project_dir / "data" / "rag_corpus" / "faiss_index"
    vectorstore.save_local(str(index_dir))
    (index_dir / "embedding_model.txt").write_text(model_name, encoding="utf-8")
    return vectorstore, embeddings, index_dir


def load_faiss_index(project_dir: Path, model_name: str | None = None, device: str | None = None):
    """Load a FAISS index created locally by build_faiss_index; the metadata sidecar is pickle-backed."""
    from langchain_community.vectorstores import FAISS

    index_dir = project_dir / "data" / "rag_corpus" / "faiss_index"
    model_name = model_name or (index_dir / "embedding_model.txt").read_text(encoding="utf-8").strip()
    embeddings = make_embeddings(model_name, device=device)
    vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    return vectorstore, embeddings


def retrieve_with_mmr(query: str, vectorstore, embeddings, k: int = 5, fetch_k: int = 20,
                      reranker: Any = None) -> list[dict[str, Any]]:
    """MMR retrieval with normalized cosine relevance scores and optional reranking."""
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    query_vector = embeddings.embed_query(query)
    import numpy as np

    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q) or 1.0
    rescored = []
    # Batch the selected chunk embeddings to avoid repeatedly initializing model work.
    document_vectors = embeddings.embed_documents([doc.page_content for doc in mmr_docs]) if mmr_docs else []
    for doc, document_vector in zip(mmr_docs, document_vectors):
        d = np.asarray(document_vector, dtype=np.float32)
        denom = (np.linalg.norm(d) * q_norm) or 1.0
        rescored.append({"document": doc, "score": float(np.dot(q, d) / denom)})
    if reranker is not None and rescored:
        scores = reranker.predict([(query, item["document"].page_content) for item in rescored])
        for item, score in zip(rescored, scores):
            item["score"] = float(1 / (1 + __import__("math").exp(-float(score))))
        rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download", action="store_true", help="Download the official PDFs before processing")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    docs = build_corpus_chunks(args.project_dir, download=args.download)
    build_faiss_index(docs, args.project_dir, model_name=args.embedding_model)


if __name__ == "__main__":
    main()
