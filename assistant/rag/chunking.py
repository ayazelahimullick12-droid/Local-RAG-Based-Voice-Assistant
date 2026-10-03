"""Loads knowledge/*.md and *.txt, splits at markdown headings, then further
splits each section into ~900-char passages, each prefixed with its heading
(so the embedding and the LLM both see which section a passage came from).
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Passage:
    id: str
    heading: str
    text: str  # heading-prefixed text sent to embedding/LLM
    content_hash: str
    source: str


def load_passages(knowledge_dir: str, chunk_chars: int = 900) -> List[Passage]:
    passages: List[Passage] = []
    root = Path(knowledge_dir)
    if not root.exists():
        return passages
    for path in sorted(root.glob("**/*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        passages.extend(_split_file(text, chunk_chars, source=str(path)))
    return passages


def _split_file(text: str, chunk_chars: int, source: str) -> List[Passage]:
    sections = _split_by_heading(text)
    passages: List[Passage] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        for chunk in _chunk_text(body, chunk_chars):
            prefixed = f"{heading}\n{chunk}" if heading else chunk
            content_hash = hashlib.sha256(prefixed.encode("utf-8")).hexdigest()
            passages.append(Passage(
                id=content_hash[:16], heading=heading, text=prefixed,
                content_hash=content_hash, source=source,
            ))
    return passages


def _split_by_heading(text: str) -> List[Tuple[str, str]]:
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return [("", text)]
    sections: List[Tuple[str, str]] = []
    if headings[0].start() > 0:
        sections.append(("", text[: headings[0].start()]))
    for i, m in enumerate(headings):
        heading = m.group(2).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _chunk_text(text: str, chunk_chars: int) -> List[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:chunk_chars]]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks
