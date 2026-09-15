from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup

from .dataset import Dataset, Document, digest, local_file

PARSER_VERSION = "html-text-v1"


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse(dataset: Dataset, document: Document, chunk_size: int = 1600) -> list[Chunk]:
    raw = local_file(dataset.root, document.path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup.find_all(["script", "style", "ix:hidden", "noscript"]):
        node.decompose()
    text = re.sub(r"[ \t]+", " ", soup.get_text(" ", strip=True)).strip()
    # A simple text parser is intentional. No OCR, semantic table reconstruction,
    # or special handling for every issuer. Originals remain available for annotation.
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        value = text[start:end]
        chunk_id = f"{document.id}:{digest([document.sha256, PARSER_VERSION, start, end])[:12]}"
        chunks.append(Chunk(chunk_id, document.id, value, start, end))
        start = end + (1 if end < len(text) and text[end] == " " else 0)
    return chunks


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def retrieve(query: str, chunks: list[Chunk], limit: int) -> list[Chunk]:
    wanted = tokens(query)
    ranked = sorted(chunks, key=lambda c: (-len(wanted & tokens(c.text)), c.id))
    return [c for c in ranked if wanted & tokens(c.text)][:limit]
