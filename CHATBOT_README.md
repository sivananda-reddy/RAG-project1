# Enterprise RAG Chatbot

A professional, enterprise-grade chatbot interface for your RAG (Retrieval-Augmented Generation) system.

## 🚀 Quick Start

### Option 1: Using the Startup Script (Easiest)

**Windows:**
```bash
# PowerShell
.\start_chatbot.ps1

# OR Command Prompt
start_chatbot.bat
```

### Option 2: Manual Start

```bash
# Install dependencies (if not already installed)
pip install streamlit streamlit-chat

# Run the chatbot
streamlit run chatbot.py
```

The chatbot will open automatically in your web browser at `http://localhost:8501`

## ✨ Features

### 🎯 Enterprise-Grade Performance
- **Persistent Embeddings**: Documents are indexed once and cached - no re-processing on every query!
- **Intelligent Caching**: Automatically detects if documents have already been indexed
- **Incremental Updates**: Only re-index when you choose to force reload

### 💬 Professional Chat Interface
- **Modern UI**: Clean, professional design suitable for enterprise use
- **Chat History**: Maintains context across multiple questions
- **Source Transparency**: View the exact document chunks used for each answer
- **Similarity Scores**: See how relevant each source document is

### 🔧 Advanced Features
- **Model Selection**: Choose from different AI models (GPT-3.5, GPT-4, Claude, etc.)
- **Temperature Control**: Adjust response creativity (0.0 = focused, 1.0 = creative)
- **Export Conversations**: Download chat history as JSON for record-keeping
- **Session Management**: Clear history or reset embeddings as needed

### 📊 Real-Time Statistics
- Document count
- Chat message history
- Pipeline status
- Embedding information

## 📖 Usage Guide

### 1. Initial Setup

1. **Configure API Key** (Already done!)
   - Your OpenRouter API key is configured in `.env`
   - Using model: `openai/gpt-3.5-turbo`

2. **Add Documents**
   - Place PDF, TXT, MD, or PY files in the `data/` folder (and subfolders like `data/doodle_labs/`)
   - For **Simple Config / Mesh Rider GUI** answers (e.g. “how to configure channel and bandwidth in Simple Config”), add the video transcript: run `python scripts/download_youtube_transcripts.py 4fUaFuf3wH0` then click **Update with New Documents**. The chatbot will then include the GUI walkthrough in retrieval for Simple Config questions.

### 2. Starting the Chatbot

1. Run the startup script or use manual start
2. The interface will open in your browser
3. Click **"🚀 Initialize Pipeline"** in the sidebar

**First Time:**
- Will load and index your PDF documents
- Takes ~30-60 seconds depending on document size

**Subsequent Times:**
- Uses cached embeddings (instant startup!)
- No re-processing needed

### 3. Chatting with Your Documents

1. Type your question in the chat input
2. Get AI-generated answers based on your documents
3. Click "📚 View Sources" to see which document sections were used
4. Continue the conversation with follow-up questions

## 🎨 Interface Overview

### Sidebar Controls

- **⚙️ Configuration**
  - Model Name: Choose your AI model
  - Temperature: Adjust response creativity
  - Force Re-index: Force document re-processing

- **📊 Pipeline Stats**
  - Documents: Number of indexed documents
  - Chat Messages: Conversation length

- **🔧 Actions**
  - Clear Chat History
  - Export Conversation
  - Clear Embeddings

### Main Chat Area

- **Chat Messages**: Conversation history with timestamps
- **Source Documents**: Expandable sections showing retrieved content
- **Similarity Scores**: Relevance metrics for each source

## 🔍 Example Queries

### For Machine Learning Documents:
- "What is machine learning?"
- "Explain supervised vs unsupervised learning"
- "What are neural networks?"
- "Describe the types of machine learning algorithms"

### For Your Specific Documents:
- "Summarize the main concepts in the document"
- "What are the key topics covered?"
- "Explain [specific topic] from the document"

## ⚙️ Configuration Options

### Model Selection

You can use different models through OpenRouter:

```bash
# Fast and cost-effective
LLM_MODEL=openai/gpt-3.5-turbo

# More capable
LLM_MODEL=openai/gpt-4

# Claude models
LLM_MODEL=anthropic/claude-3-haiku
LLM_MODEL=anthropic/claude-3-sonnet

# Open source models
LLM_MODEL=meta-llama/llama-3-8b-instruct
```

### Temperature Settings

- **0.0 - 0.3**: Very focused, factual responses
- **0.4 - 0.7**: Balanced creativity and accuracy (recommended)
- **0.8 - 1.0**: More creative, exploratory responses

## 🛠️ Troubleshooting

### Chatbot Won't Start
```bash
# Install dependencies
pip install -r requirements.txt
```

### Embeddings Not Loading
- Check if PDFs are in the `data/` folder
- Try clicking "Clear Embeddings" and re-initialize
- Check console for error messages

### API Errors
- Verify your OpenRouter API key in `.env`
- Check your OpenRouter account balance
- Try a different model

### Slow First Load
- First-time indexing takes time (normal behavior)
- Subsequent loads will be instant
- Consider using a smaller chunk size for faster processing

## 📈 Performance Tips

1. **Optimize Chunk Size**: For large documents, smaller chunks (300-400) work better
2. **Use Appropriate Model**: GPT-3.5-turbo is fastest and most cost-effective
3. **Manage History**: Clear chat history periodically for better context
4. **Export Regularly**: Download conversations for record-keeping

## 🔒 Security & Privacy

- **Local Processing**: Embeddings generated locally (no data sent for indexing)
- **API Security**: API keys stored in local `.env` file
- **Data Privacy**: Documents stay on your machine
- **Session Management**: Chat history stored in browser session

## 📝 Architecture

```
Enterprise RAG Chatbot
│
├── chatbot.py                 # Main Streamlit application
├── src/
│   ├── optimized_rag.py       # Persistent embeddings logic
│   ├── rag_chain.py           # Core RAG implementation
│   ├── pdf_loader.py          # Document processing
│   ├── embeddings.py          # Embedding generation
│   └── vector_store.py        # ChromaDB management
│
├── data/                      # PDF documents
├── chroma_db/                 # Persistent vector database
├── .env                       # Configuration
└── requirements.txt           # Dependencies
```

## 🆘 Getting Help

1. **Check Console**: Look for error messages in the terminal
2. **Review Logs**: Streamlit shows detailed error information
3. **Restart Server**: Sometimes a fresh start helps
4. **Check Dependencies**: Ensure all packages are installed

## 🎯 Best Practices

1. **Document Organization**
   - Keep related documents together
   - Use descriptive filenames
   - Remove outdated documents

2. **Query Formulation**
   - Be specific in your questions
   - Ask follow-up questions for clarity
   - Use context from previous answers

3. **Performance Management**
   - Clear embeddings when adding new documents
   - Export important conversations
   - Monitor API usage

## 🚀 Production Deployment

For production use:

1. **Use Production Server**
   ```bash
   streamlit run chatbot.py --server.headless=true
   ```

2. **Configure Authentication**
   - Add Streamlit authentication
   - Use environment variables for API keys

3. **Monitor Usage**
   - Track API calls
   - Monitor response times
   - Log user interactions

---

**Built with ❤️ using Streamlit, LangChain, and ChromaDB**
