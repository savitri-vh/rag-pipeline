"""
Step 1: Load documents.

Supports plain text (.txt, .md), PDFs (.pdf), and a generic in-memory
loader for scraped/raw strings. Every loaded document is normalized into
a `Document` dataclass carrying its text plus a source identifier, so
later stages (and final citations) can always point back to where a
passage came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str          # stable identifier, e.g. filename
    text: str            # full raw text of the document
    metadata: dict = field(default_factory=dict)


def load_text_file(path: str) -> Document:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return Document(doc_id=p.name, text=text, metadata={"path": str(p), "type": "text"})


def load_pdf_file(path: str) -> Document:
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Tag each page so chunks can cite a page number later.
        pages_text.append(f"\n[page {i + 1}]\n{page_text}")
    full_text = "".join(pages_text)
    return Document(doc_id=p.name, text=full_text, metadata={"path": str(p), "type": "pdf", "num_pages": len(reader.pages)})


def load_raw_string(doc_id: str, text: str, metadata: dict | None = None) -> Document:
    """For scraped data / API responses / any in-memory text."""
    return Document(doc_id=doc_id, text=text, metadata=metadata or {"type": "raw"})


def load_directory(dir_path: str, extensions=(".txt", ".md", ".pdf")) -> List[Document]:
    """Load every supported file in a directory (non-recursive by default)."""
    docs: List[Document] = []
    for name in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in extensions:
            continue
        try:
            if ext == ".pdf":
                docs.append(load_pdf_file(full))
            else:
                docs.append(load_text_file(full))
        except Exception as e:
            print(f"[loader] skipping {name}: {e}")
    return docs
