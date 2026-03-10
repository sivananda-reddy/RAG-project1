"""
Unified Document Loading Module

Loads PDF, TXT, and Markdown (.md) files from a data directory.
"""

import os
from pathlib import Path
from typing import List, Dict
import logging

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".py"}


class DocumentLoader:
    """Loads PDF, TXT, and Markdown documents from a directory."""

    def __init__(self, data_path: str = "./data"):
        """
        Initialize document loader.

        Args:
            data_path: Path to directory containing documents
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, file_path: Path) -> Dict:
        """Load a single PDF file."""
        try:
            if not PdfReader:
                raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return {"filename": file_path.name, "text": "", "metadata": {"source": file_path.name}, "error": "File not found"}

            reader = PdfReader(str(file_path))
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text() or ""

            logger.info(f"Successfully loaded PDF: {file_path.name}")
            return {
                "filename": file_path.name,
                "text": text.strip(),
                "metadata": {"source": file_path.name, "num_pages": len(reader.pages), "type": "pdf"}
            }
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return {"filename": file_path.name, "text": "", "metadata": {"source": file_path.name}, "error": str(e)}

    def load_text_file(self, file_path: Path, encoding: str = "utf-8") -> Dict:
        """Load a TXT or Markdown file."""
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return {"filename": file_path.name, "text": "", "metadata": {"source": file_path.name}, "error": "File not found"}

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                text = f.read()

            suffix = file_path.suffix.lower()
            if suffix == ".md":
                file_type = "markdown"
            elif suffix == ".py":
                file_type = "python"
            else:
                file_type = "text"
            logger.info(f"Successfully loaded {file_type} file: {file_path.name}")
            return {
                "filename": file_path.name,
                "text": text.strip(),
                "metadata": {"source": file_path.name, "type": file_type}
            }
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return {"filename": file_path.name, "text": "", "metadata": {"source": file_path.name}, "error": str(e)}

    def load_file(self, file_path: Path) -> Dict:
        """
        Load a single file (PDF, TXT, or MD) based on extension.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with filename, text, and metadata
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(file_path)
        if suffix in (".txt", ".md", ".py"):
            return self.load_text_file(file_path)
        logger.warning(f"Unsupported file type: {file_path.name}")
        return {"filename": file_path.name, "text": "", "metadata": {"source": file_path.name}, "error": f"Unsupported extension: {suffix}"}

    def get_supported_files(self) -> List[Path]:
        """Get list of all supported files (PDF, TXT, MD, PY) in the data directory and subdirectories."""
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(self.data_path.glob(f"**/*{ext}"))
        return sorted(files)

    def load_all_documents(self) -> List[Dict]:
        """
        Load all supported documents (PDF, TXT, MD, PY) from the data directory.

        Returns:
            List of dictionaries with filename, text, and metadata
        """
        documents = []
        files = self.get_supported_files()

        if not files:
            logger.warning(f"No supported files (PDF, TXT, MD, PY) found in {self.data_path}")
            return []

        logger.info(f"Found {len(files)} document(s): PDF, TXT, MD, PY")

        for file_path in files:
            doc = self.load_file(file_path)
            if "error" not in doc and doc.get("text", "").strip():
                documents.append(doc)

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text = text.strip()

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks


# Backward compatibility: PDFLoader-style interface
def load_all_pdfs_compat(loader: DocumentLoader) -> List[Dict]:
    """Return all loaded documents (same as load_all_documents). Kept for compatibility."""
    return loader.load_all_documents()


if __name__ == "__main__":
    loader = DocumentLoader("./data")
    documents = loader.load_all_documents()

    print(f"\nLoaded {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['filename']}: {len(doc['text'])} chars (type: {doc.get('metadata', {}).get('type', 'unknown')})")
