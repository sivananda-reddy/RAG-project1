"""
Embeddings Generation Module

This module handles creating vector embeddings from text.
Supports both local (sentence-transformers) and OpenAI embeddings.
"""

import os
import logging
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings from text."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "local"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model
            embedding_type: Type of embeddings ('local' or 'openai')
        """
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.model = None
        
        if embedding_type == "local":
            self._init_local_embeddings()
        elif embedding_type == "openai":
            self._init_openai_embeddings()
    
    def _init_local_embeddings(self):
        """Initialize local sentence-transformer model."""
        if not SentenceTransformer:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Loading local embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _init_openai_embeddings(self):
        """Initialize OpenAI embeddings."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Set it in .env file or environment variables"
            )
        
        try:
            from langchain.embeddings.openai import OpenAIEmbeddings
            self.model = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=self.model_name
            )
            logger.info("OpenAI embedding model initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            if self.embedding_type == "local":
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            else:  # openai
                embedding = self.model.embed_query(text)
                return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        if not texts:
            return []
        
        try:
            if self.embedding_type == "local":
                embeddings = self.model.encode(texts, batch_size=batch_size)
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            else:  # openai
                embeddings = [self.model.embed_query(text) for text in texts]
                return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.embedding_type == "local":
            if self.model_name == "all-MiniLM-L6-v2":
                return 384
            elif self.model_name == "all-mpnet-base-v2":
                return 768
            else:
                # Generate a sample embedding to determine dimension
                sample_embedding = self.model.encode("test")
                return len(sample_embedding)
        else:  # openai
            return 1536  # OpenAI embedding dimension
    
    @staticmethod
    def similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator(embedding_type="local")
    
    # Single text embedding
    text = "Machine learning is a subset of artificial intelligence"
    embedding = generator.embed_text(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    
    # Multiple texts
    texts = [
        "Machine learning and artificial intelligence",
        "Deep learning with neural networks",
        "Natural language processing"
    ]
    embeddings = generator.embed_texts(texts)
    print(f"\nGenerated {len(embeddings)} embeddings")
    
    # Similarity calculation
    sim = generator.similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 1 and 2: {sim:.4f}")
