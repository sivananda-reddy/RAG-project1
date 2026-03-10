"""
Enterprise RAG Chatbot with Streamlit

A professional chatbot interface for the RAG system with:
- Persistent embeddings (no re-embedding on every query)
- Chat history management
- Professional UI
- Export conversations
"""

import streamlit as st
from streamlit_chat import message
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.optimized_rag import OptimizedRAGPipeline

# Page config
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Metrics */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Sources expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pipeline_loaded' not in st.session_state:
        st.session_state.pipeline_loaded = False
    
    if 'embedding_status' not in st.session_state:
        st.session_state.embedding_status = None
    
    if 'embeddings_cleared' not in st.session_state:
        st.session_state.embeddings_cleared = False
    
    # Try to auto-load if embeddings exist (skip if user just cleared embeddings)
    if not st.session_state.pipeline_loaded and st.session_state.rag_pipeline is None and not st.session_state.embeddings_cleared:
        try_auto_load()


@st.cache_resource
def get_pipeline():
    """Create and cache the RAG pipeline (survives page refresh)."""
    return OptimizedRAGPipeline(
        embeddings_type="local",
        llm_type="openai",
        model_name=st.session_state.get('model_name', 'openai/gpt-3.5-turbo'),
        temperature=st.session_state.get('temperature', 0.7)
    )


def try_auto_load():
    """Try to automatically load pipeline if embeddings exist."""
    try:
        pipeline = get_pipeline()
        
        # Check if embeddings already exist
        if pipeline.check_embeddings_exist():
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_loaded = True
            st.session_state.embedding_status = {
                'message': 'Loaded existing embeddings',
                'skipped': True
            }
    except Exception as e:
        # Silently fail - user can manually initialize
        pass


def load_pipeline():
    """Load the RAG pipeline."""
    with st.spinner("Initializing RAG pipeline..."):
        try:
            pipeline = get_pipeline()
            
            # Load documents with caching
            status = pipeline.load_documents_incremental(
                force_reload=st.session_state.get('force_reload', False)
            )
            
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_loaded = True
            st.session_state.embedding_status = status
            
            return True
        except Exception as e:
            st.error(f"Error loading pipeline: {str(e)}")
            return False


def update_embeddings():
    """Update embeddings with new documents."""
    with st.spinner("Scanning for new documents and updating embeddings..."):
        try:
            pipeline = st.session_state.rag_pipeline or get_pipeline()
            
            # Force reload to pick up new documents
            status = pipeline.load_documents_incremental(force_reload=True)
            
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_loaded = True
            st.session_state.embedding_status = status
            
            return True
        except Exception as e:
            st.error(f"Error updating embeddings: {str(e)}")
            return False


def display_chat_message(message_obj):
    """Display a chat message."""
    with st.chat_message(message_obj["role"]):
        st.markdown(message_obj["content"])

        # Show retrieval mode for assistant messages
        if message_obj["role"] == "assistant" and message_obj.get("retrieval_used"):
            ru = message_obj["retrieval_used"]
            if ru.get("hybrid"):
                st.caption(f"🔀 Vector ({ru.get('vector_chunks', 0)} chunks) + Graph ({ru.get('graph_triples', 0)} triples)")
            else:
                st.caption(f"📄 Vector ({ru.get('vector_chunks', 0)} chunks)")

        # Show sources for assistant messages
        if message_obj["role"] == "assistant" and "sources" in message_obj:
            with st.expander("📚 View Sources", expanded=False):
                for idx, source in enumerate(message_obj["sources"], 1):
                    st.caption(f"**Source {idx}** (Similarity: {source['similarity']:.3f})")
                    st.text(source['text'])
                    st.markdown(f"*From: {source['metadata'].get('source', 'Unknown')}*")
                    st.divider()


def process_query(query: str):
    """Process a user query."""
    if not st.session_state.pipeline_loaded:
        st.error("Pipeline not loaded. Please initialize first.")
        return
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_pipeline.chat(query)
                
                st.markdown(result['answer'])
                
                # Show retrieval mode (vector vs hybrid)
                ru = result.get('retrieval_used', {})
                if ru:
                    if ru.get('hybrid'):
                        st.caption(f"🔀 **Retrieval:** Vector ({ru.get('vector_chunks', 0)} chunks) + Knowledge graph ({ru.get('graph_triples', 0)} triples)")
                    else:
                        st.caption(f"📄 **Retrieval:** Vector ({ru.get('vector_chunks', 0)} chunks)")
                
                # Show sources
                with st.expander("📚 View Sources", expanded=False):
                    for idx, source in enumerate(result['sources'], 1):
                        st.caption(f"**Source {idx}** (Similarity: {source['similarity']:.3f})")
                        st.text(source['text'])
                        st.markdown(f"*From: {source['metadata'].get('source', 'Unknown')}*")
                        st.divider()
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources'],
                    "retrieval_used": result.get('retrieval_used'),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })


def export_conversation():
    """Export conversation to JSON."""
    if not st.session_state.messages:
        st.warning("No messages to export.")
        return
    
    conversation = {
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    
    json_str = json.dumps(conversation, indent=2)
    st.download_button(
        label="📥 Export Conversation",
        data=json_str,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def main():
    """Main application."""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Enterprise RAG Chatbot")
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        model_name = st.text_input(
            "Model Name",
            value=st.session_state.get('model_name', 'openai/gpt-3.5-turbo'),
            help="OpenRouter model name (e.g., openai/gpt-3.5-turbo, anthropic/claude-3-haiku)"
        )
        st.session_state.model_name = model_name
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.7),
            step=0.1,
            help="Higher values = more creative responses"
        )
        st.session_state.temperature = temperature
        
        force_reload = st.checkbox(
            "Force Re-index Documents",
            value=False,
            help="Check this to re-embed all documents (slower)"
        )
        st.session_state.force_reload = force_reload
        
        st.markdown("---")
        
        # Initialize button
        if st.button("🚀 Initialize Pipeline", type="primary"):
            st.session_state.embeddings_cleared = False
            with st.spinner("Loading..."):
                if load_pipeline():
                    st.success("✅ Pipeline initialized successfully!")
        
        # Update with New Documents – important when only one doc is indexed
        indexed_count = 0
        if st.session_state.pipeline_loaded and st.session_state.rag_pipeline:
            indexed_count = len(st.session_state.rag_pipeline.get_stats().get('indexed_sources', []))
        if indexed_count <= 1:
            st.warning("Only 1 document is indexed. To search **sense_2026** and other files in `data/`, click **Update with New Documents** below. This may take 1–2 min.")
        if st.button("🔄 Update with New Documents", type="primary" if indexed_count <= 1 else "secondary"):
            with st.spinner("Scanning data/ and subfolders (PDF, TXT, MD, PY)... This may take 1–2 minutes."):
                if update_embeddings():
                    st.success("✅ All documents re-indexed! Check **Indexed documents** below to see all files. You can now query sense and other files.")
        
        # Status indicator
        if st.session_state.pipeline_loaded:
            st.success("✅ Pipeline Ready")
        else:
            st.info("ℹ️ Click Initialize to start")
        
        st.markdown("---")
        
        # Stats (hide when user just cleared embeddings so list never shows stale data)
        if (st.session_state.pipeline_loaded and st.session_state.rag_pipeline
                and not st.session_state.get('embeddings_cleared', False)):
            st.subheader("📊 Pipeline Stats")
            stats = st.session_state.rag_pipeline.get_stats()
            doc_count = stats['vector_store'].get('count', 0)
            indexed_sources = stats.get('indexed_sources') or []
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_count)
            with col2:
                st.metric("Chat Messages", stats['chat_history_length'])
            if stats.get('retrieval_mode'):
                st.caption(f"🔀 {stats['retrieval_mode']}")
            if stats.get('knowledge_graph_triples') is not None:
                st.metric("Knowledge graph triples", stats['knowledge_graph_triples'])
            # Only show Indexed documents when we actually have indexed docs (hide when 0 after clear)
            if indexed_sources and doc_count > 0:
                only_one = len(indexed_sources) <= 1
                with st.expander("📁 Indexed documents", expanded=only_one):
                    for name in indexed_sources:
                        st.caption(f"• {name}")
                    if only_one:
                        st.warning("Only this file is searchable. To add **sense_2026** and other files, click **Update with New Documents** above.")
                    else:
                        st.caption("Queries search across all of these. To add new files, click **Update with New Documents**.")
            st.markdown("---")
        
        # Actions
        st.subheader("🔧 Actions")
        
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_chat_history()
                st.session_state.messages = []
                st.success("Chat history cleared!")
        
        export_conversation()
        
        if st.button("🔄 Clear Embeddings"):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_embeddings()
                st.session_state.rag_pipeline = None
                st.session_state.pipeline_loaded = False
                st.session_state.embedding_status = None
                st.session_state.embeddings_cleared = True
                get_pipeline.clear()
                st.success("Embeddings cleared! Re-initialize to re-index.")
                st.rerun()
    
    # Main chat area
    st.title("💬 Chat with your Documents")
    
    # Status banner
    if st.session_state.embedding_status:
        status = st.session_state.embedding_status
        if status['skipped']:
            st.info(f"ℹ️ {status['message']}")
        else:
            st.success(f"✅ {status['message']}")
    
    # Chat interface
    if not st.session_state.pipeline_loaded:
        st.info("👈 Please initialize the pipeline from the sidebar to start chatting.")
        
        # Show instructions
        with st.expander("📖 Getting Started", expanded=True):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Configure Settings** (Optional)
               - Adjust model name and temperature in the sidebar
               - Default settings work well for most use cases
            
            2. **Initialize Pipeline**
               - Click "🚀 Initialize Pipeline" in the sidebar
               - First time: Will load and index your PDF documents
               - Subsequent times: Will use cached embeddings (fast!)
            
            3. **Start Chatting**
               - Ask questions about your documents
               - View sources for transparency
               - Export conversations for record-keeping
            
            ### Features
            
            - ✅ **Persistent Embeddings**: No re-processing on every query
            - ✅ **Professional UI**: Clean, enterprise-grade interface
            - ✅ **Source Transparency**: See which documents were used
            - ✅ **Chat History**: Context maintained across questions
            - ✅ **Export**: Download conversations as JSON
            """)
    else:
        # Display existing messages
        for msg in st.session_state.messages:
            display_chat_message(msg)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            process_query(prompt)


if __name__ == "__main__":
    main()
