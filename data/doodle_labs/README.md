# Doodle Labs Technical Library (downloaded)

This folder is populated by the download scripts. Do not edit files here manually.

## How to populate

### 1. Web pages and PDFs

From the project root:

```bash
pip install beautifulsoup4 requests
python scripts/download_doodle_labs.py
```

This will:

1. Crawl https://techlibrary.doodlelabs.com/ (Technical Library)
2. Save each documentation page as a `.txt` file
3. Download any linked `.pdf` files from the same site

### 2. YouTube video transcripts (GUI walkthrough, etc.)

Video content (e.g. Mesh Rider GUI Quick Walkthrough) is only available as video; the scripts turn it into text so the RAG can use it:

```bash
pip install youtube-transcript-api
python scripts/download_youtube_transcripts.py --crawl
```

Or download a specific video:

```bash
python scripts/download_youtube_transcripts.py "https://www.youtube.com/watch?v=4fUaFuf3wH0"
```

Transcripts are saved as e.g. `mesh-rider-gui-walkthrough-4fUaFuf3wH0.txt`.

## After running

Click **Update with New Documents** in the RAG chatbot to index these files.

## Source

- [Doodle Labs Technical Library](https://techlibrary.doodlelabs.com/doodle-labs-technical-library)
