Place your documents here for processing.

**Supported file types:**
- **PDF** (`.pdf`) – Research papers, reports, books, articles
- **Text** (`.txt`) – Plain text files
- **Markdown** (`.md`) – Markdown documentation, notes, READMEs
- **Python** (`.py`) – Python source code

The RAG system will:
1. Load all supported files from this directory
2. Extract text content
3. Create embeddings
4. Store in the vector database
5. Enable semantic search and chat

**Tips:**
- Use UTF-8 encoding for `.txt`, `.md`, and `.py` files
- Text-based PDFs work best (scanned PDFs may require OCR)
- Add or remove files anytime, then click **Update with New Documents** in the chatbot to re-index
