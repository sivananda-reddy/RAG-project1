"""
Utility Functions

Helper functions for the RAG project.
"""

import logging
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_results(question: str, answer: str, sources: List[Dict]) -> None:
    """
    Pretty print RAG results.
    
    Args:
        question: The question asked
        answer: The answer generated
        sources: List of source documents used
    """
    print("\n" + "="*70)
    print(f"QUESTION: {question}")
    print("="*70)
    print(f"\nANSWER:\n{answer}")
    print("\n" + "-"*70)
    print("SOURCES:")
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. Similarity: {source.get('similarity', 0):.4f}")
        print(f"   Source: {source.get('metadata', {}).get('source', 'Unknown')}")
        print(f"   Text: {source['text'][:100]}...")
    print("="*70 + "\n")


def save_results(results: List[Dict], output_file: str = "results.json") -> None:
    """
    Save RAG results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def load_config(config_file: str = ".env") -> Dict:
    """
    Load configuration from .env file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Dictionary of configuration values
    """
    from dotenv import dotenv_values
    try:
        config = dotenv_values(config_file)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}


def validate_documents(documents: List[Dict]) -> bool:
    """
    Validate document structure.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        True if all documents are valid
    """
    if not documents:
        logger.warning("Document list is empty")
        return False
    
    for i, doc in enumerate(documents):
        if 'text' not in doc or not doc['text'].strip():
            logger.warning(f"Document {i} has empty or missing text")
            return False
    
    return True


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Number of characters to overlap
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def calculate_statistics(embeddings: List[List[float]]) -> Dict:
    """
    Calculate statistics about embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Dictionary with statistics
    """
    import numpy as np
    
    if not embeddings:
        return {}
    
    embeddings_array = np.array(embeddings)
    
    return {
        'num_embeddings': len(embeddings),
        'dimension': len(embeddings[0]) if embeddings else 0,
        'mean': float(np.mean(embeddings_array)),
        'std': float(np.std(embeddings_array)),
        'min': float(np.min(embeddings_array)),
        'max': float(np.max(embeddings_array))
    }


class ConsoleFormatter:
    """Utility for formatting console output."""
    
    @staticmethod
    def header(text: str, width: int = 70) -> str:
        """Format text as a header."""
        return "\n" + "="*width + "\n" + text.center(width) + "\n" + "="*width + "\n"
    
    @staticmethod
    def subheader(text: str, width: int = 70) -> str:
        """Format text as a subheader."""
        return "\n" + "-"*width + "\n" + text + "\n" + "-"*width + "\n"
    
    @staticmethod
    def success(text: str) -> str:
        """Format success message."""
        return f"✓ {text}"
    
    @staticmethod
    def error(text: str) -> str:
        """Format error message."""
        return f"✗ {text}"
    
    @staticmethod
    def info(text: str) -> str:
        """Format info message."""
        return f"ℹ {text}"


if __name__ == "__main__":
    # Test utilities
    print(ConsoleFormatter.header("RAG Project Utilities"))
    print(ConsoleFormatter.success("Utilities module loaded"))
    
    # Test chunking
    sample_text = "This is a test. " * 50
    chunks = chunk_text(sample_text, chunk_size=100, overlap=10)
    print(f"\nText chunking: {len(chunks)} chunks created")
