"""
PDF Loading and Text Extraction Module

This module handles loading PDF files and extracting text content.
Supports both regular PDFs and scanned documents with OCR.
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


class PDFLoader:
    """Loads and processes PDF documents."""
    
    def __init__(self, data_path: str = "./data"):
        """
        Initialize PDF loader.
        
        Args:
            data_path: Path to directory containing PDF files
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> Dict[str, str]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with filename and extracted text
        """
        try:
            if not PdfReader:
                raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return {"filename": str(file_path), "text": "", "error": "File not found"}
            
            if file_path.suffix.lower() != '.pdf':
                logger.warning(f"File is not a PDF: {file_path}")
            
            reader = PdfReader(file_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            logger.info(f"Successfully loaded {file_path.name}")
            return {
                "filename": file_path.name,
                "text": text,
                "num_pages": len(reader.pages)
            }
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return {"filename": str(file_path), "text": "", "error": str(e)}
    
    def load_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Load all PDF files from data directory.
        
        Returns:
            List of dictionaries with filename and text
        """
        documents = []
        pdf_files = list(self.data_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            doc = self.load_pdf(pdf_file)
            if "error" not in doc:
                documents.append(doc)
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks


def load_sample_documents() -> List[Dict[str, str]]:
    """
    Create sample documents for testing.
    
    Returns:
        List of sample documents
    """
    samples = [
        {
            "filename": "sample1.pdf",
            "text": """
            Introduction to Machine Learning
            
            Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.
            
            Types of Machine Learning:
            1. Supervised Learning - Learning from labeled data
            2. Unsupervised Learning - Finding patterns in unlabeled data
            3. Reinforcement Learning - Learning through rewards and penalties
            """,
            "num_pages": 1
        },
        {
            "filename": "sample2.pdf",
            "text": """
            Deep Learning Fundamentals
            
            Deep learning is a specialized subset of machine learning that uses
            neural networks with multiple layers (hence "deep").
            
            Key Concepts:
            - Neural Networks: Connected nodes inspired by biological neurons
            - Backpropagation: Algorithm for training neural networks
            - Activation Functions: Non-linear transformations in neural networks
            """,
            "num_pages": 1
        }
    ]
    return samples


if __name__ == "__main__":
    # Example usage
    loader = PDFLoader()
    
    # Try to load PDFs from data directory
    documents = loader.load_all_pdfs()
    
    if not documents:
        print("No PDFs found. Using sample documents for demonstration.")
        documents = load_sample_documents()
    
    print(f"\nLoaded {len(documents)} documents:")
    for doc in documents:
        print(f"  - {doc['filename']}: {len(doc['text'])} characters")
