"""
RAG Chain Implementation

This module implements the complete Retrieval-Augmented Generation pipeline.
"""

import os
import logging
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

try:
    from langchain.chains import RetrievalQA
    from langchain_community.chat_models import ChatOpenAI
    from langchain.llms.fake import FakeListLLM
except ImportError:
    RetrievalQA = None

from document_loader import DocumentLoader
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from project root
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)


class RAGPipeline:
    """Implements the RAG pipeline."""
    
    def __init__(
        self,
        embeddings_type: str = "local",
        llm_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embeddings_type: Type of embeddings ('local' or 'openai')
            llm_type: Type of LLM ('openai' or 'fake')
            model_name: Name of the LLM model
            temperature: Temperature for LLM generation
        """
        self.embeddings_type = embeddings_type
        self.embeddings = EmbeddingGenerator(embedding_type=embeddings_type)
        self.vector_store = VectorStore()
        self.llm = self._init_llm(llm_type, model_name, temperature)
        
        logger.info("RAG pipeline initialized")
    
    def _init_llm(self, llm_type: str, model_name: str, temperature: float):
        """Initialize the LLM."""
        if llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set. Using fake LLM for demonstration."
                )
                return FakeListLLM(responses=[
                    "This is a demo response. To use real AI responses, please set your OPENAI_API_KEY in the .env file."
                ])
            
            try:
                # Check if using OpenRouter (key starts with sk-or-)
                api_base = os.getenv("OPENAI_API_BASE")
                
                # Prepare ChatOpenAI parameters
                llm_params = {
                    "openai_api_key": api_key,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_retries": 2
                }
                
                # Add base URL if provided (for OpenRouter or other providers)
                if api_base:
                    llm_params["openai_api_base"] = api_base
                    logger.info(f"Using custom API base: {api_base}")
                
                llm = ChatOpenAI(**llm_params)
                
                # Test the API key with a simple call
                try:
                    from langchain.schema import HumanMessage
                    test_msg = [HumanMessage(content="test")]
                    llm.invoke(test_msg)
                    logger.info("API key validated successfully")
                    return llm
                except Exception as api_error:
                    error_str = str(api_error)
                    if "429" in error_str or "insufficient_quota" in error_str:
                        logger.warning(
                            "API quota exceeded. Using demo LLM. "
                            "Please check your billing."
                        )
                        return FakeListLLM(responses=[
                            "This is a demo response. Your API quota has been exceeded. "
                            "The retrieval system is working perfectly - it found relevant documents from your PDF. "
                            "To get real AI-generated answers, please add credits to your account."
                        ])
                    else:
                        logger.error(f"API validation error: {error_str}")
                        raise api_error
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                return FakeListLLM(responses=[
                    "Demo mode: The retrieval system found relevant documents from your PDF. "
                    "Please check your API key and billing for real AI responses."
                ])
        
        else:  # fake
            return FakeListLLM(responses=[
                "This is a demo response from the fake LLM. "
                "In production, this would be a real response from OpenAI or another LLM."
            ])
    
    def load_documents(self, pdf_path: str = None) -> None:
        """
        Load PDF documents and create embeddings.
        
        Args:
            pdf_path: Path to directory containing PDFs. If None, uses default ./data
        """
        logger.info("Loading documents...")
        
        # Use project root data directory if no path specified
        if pdf_path is None:
            pdf_path = str(PROJECT_ROOT / "data")
        elif not os.path.isabs(pdf_path):
            # If relative path, make it relative to project root
            pdf_path = str(PROJECT_ROOT / pdf_path)
        
        logger.info(f"Looking for documents (PDF, TXT, MD, PY) in: {pdf_path}")
        
        # Load PDFs
        loader = DocumentLoader(pdf_path)
        documents = loader.load_all_documents()
        
        if not documents:
            logger.warning("No documents loaded")
            return
        
        # Split documents into chunks
        chunks = []
        for doc in documents:
            text_chunks = loader.split_text(doc['text'], chunk_size=500, overlap=50)
            for chunk in text_chunks:
                chunks.append({
                    'text': chunk,
                    'metadata': {'source': doc['filename']}
                })
        
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)
        
        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_store.add_documents(chunks, embeddings)
        
        logger.info("Document loading complete")
    
    def query(self, question: str, k: int = 3) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing query: {question}")
        
        # Generate embedding for question
        question_embedding = self.embeddings.embed_text(question)
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(question_embedding, k=k)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return {
                'answer': 'No relevant documents found in the knowledge base.',
                'sources': []
            }
        
        # Prepare context
        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            # Handle different LLM types
            if hasattr(self.llm, 'invoke'):
                # ChatOpenAI or similar
                result = self.llm.invoke(messages)
                answer = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(self.llm, 'predict'):
                # Older LangChain LLMs
                answer = self.llm.predict(prompt)
            elif hasattr(self.llm, 'responses'):
                # FakeListLLM
                answer = self.llm.responses[0]
            else:
                answer = "Unable to generate response"
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "Error generating response"
        
        return {
            'answer': answer,
            'sources': [
                {
                    'text': doc['text'][:100] + '...',
                    'similarity': doc['similarity'],
                    'metadata': doc['metadata']
                }
                for doc in relevant_docs
            ]
        }
    
    def get_pipeline_info(self) -> dict:
        """Get information about the RAG pipeline."""
        collection_info = self.vector_store.get_collection_info()
        return {
            'embeddings_type': self.embeddings_type,
            'embedding_dimension': self.embeddings.get_embedding_dimension(),
            'vector_store': collection_info,
            'llm_type': type(self.llm).__name__
        }


def main():
    """Main example usage."""
    # Initialize RAG pipeline
    rag = RAGPipeline(embeddings_type="local", llm_type="openai")
    
    # Load documents
    rag.load_documents()
    
    # Query
    question = "What is machine learning?"
    result = rag.query(question)
    
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)
    print(f"Answer: {result['answer']}")
    print("\nSources:")
    for source in result['sources']:
        print(f"  - {source['text']}")
        print(f"    Similarity: {source['similarity']:.4f}")


if __name__ == "__main__":
    main()
