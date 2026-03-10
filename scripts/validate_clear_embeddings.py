"""
Validate that after clearing embeddings, indexed_sources and UI state show no documents.
Run from project root: python scripts/validate_clear_embeddings.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    from vector_store import VectorStore
    from optimized_rag import OptimizedRAGPipeline

    # Use a temp DB so we don't wipe real chroma_db
    import tempfile
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test_chroma")

    print("1. Create vector store and add one doc...")
    store = VectorStore(db_path=db_path, collection_name="test_rag")
    store.add_documents(
        [{"text": "test", "metadata": {"source": "test.txt"}}],
        [[0.1] * 384]  # dummy embedding; dimension may vary
    )
    info = store.get_collection_info()
    sources = store.get_indexed_sources()
    assert info.get("count", 0) == 1, f"Expected count 1, got {info}"
    assert len(sources) == 1, f"Expected 1 source, got {sources}"
    print(f"   count={info.get('count')}, indexed_sources={sources} OK")

    print("2. Delete collection...")
    store.delete_collection()

    print("3. New store (same path) -> get_or_create_collection creates empty collection...")
    store2 = VectorStore(db_path=db_path, collection_name="test_rag")
    info2 = store2.get_collection_info()
    sources2 = store2.get_indexed_sources()
    assert info2.get("count", 0) == 0, f"Expected count 0 after delete, got {info2}"
    assert sources2 == [], f"Expected no sources after delete, got {sources2}"
    print(f"   count={info2.get('count')}, indexed_sources={sources2} OK")

    print("4. get_stats() with count 0 returns indexed_sources=[]...")
    # Pipeline with empty store (use real chroma path for pipeline init; we only test get_stats logic)
    # We test optimized_rag.get_stats logic: when count==0, indexed_sources must be []
    class FakeStore:
        def get_collection_info(self):
            return {"count": 0, "document_count": 0}
        def get_indexed_sources(self):
            return ["stale.txt"]  # simulate stale data
    pipeline = OptimizedRAGPipeline(embeddings_type="local", llm_type="openai")
    pipeline.vector_store = FakeStore()
    stats = pipeline.get_stats()
    assert stats.get("indexed_sources") == [], f"Expected [] when count=0, got {stats.get('indexed_sources')}"
    print(f"   get_stats() indexed_sources={stats.get('indexed_sources')} OK")

    print("\nAll validations passed. Clear-embeddings flow will show no indexed documents.")
    return 0

if __name__ == "__main__":
    try:
        exit(main() or 0)
    except Exception as e:
        print(f"Validation failed: {e}")
        raise
