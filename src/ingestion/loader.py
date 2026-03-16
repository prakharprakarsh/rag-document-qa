"""Document loading — supports PDF, TXT, and DOCX files."""

from pathlib import Path
from langchain.schema import Document
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and return a list of Documents (one per page)."""
    reader = PdfReader(str(file_path))
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": i + 1},
                )
            )
    return documents


def load_text(file_path: str | Path) -> list[Document]:
    """Load a plain text file."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def load_document(file_path: str | Path) -> list[Document]:
    """Auto-detect file type and load accordingly."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_text,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    return loader(path)


def load_directory(dir_path: str | Path) -> list[Document]:
    """Load all supported documents from a directory."""
    path = Path(dir_path)
    all_docs = []
    supported = {".pdf", ".txt", ".md"}

    for file in sorted(path.iterdir()):
        if file.suffix.lower() in supported:
            try:
                docs = load_document(file)
                all_docs.extend(docs)
                print(f"  Loaded {file.name}: {len(docs)} chunk(s)")
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")

    return all_docs