# Scripts

## download_doodle_labs.py

Downloads all documentation from the **Doodle Labs Technical Library** into `data/doodle_labs/` so the RAG can index it.

### Prerequisites

```bash
pip install beautifulsoup4 requests
```

### Run

From project root:

```bash
python scripts/download_doodle_labs.py
```

- **Output:** `data/doodle_labs/*.txt` (and any `.pdf` files found on the site)
- **Rate limit:** ~1.5 s between requests
- **Limit:** Up to 200 pages per run (configurable in the script)

### After running

In the RAG chatbot, click **Update with New Documents** to index the new files. You can then ask questions about Mesh Rider, SENSE, integration guides, etc.

---

## download_youtube_transcripts.py

Downloads **YouTube video transcripts** linked from the Doodle Labs Technical Library and saves them as `.txt` in `data/doodle_labs/`. The RAG can then answer from video content (e.g. GUI walkthrough, Simple Config steps).

**Source:** Video links are listed at [Doodle Labs Technical Library](https://techlibrary.doodlelabs.com/doodle-labs-technical-library) (e.g. GUI Quick Walkthrough Guide (Video)).

### Prerequisites

```bash
pip install youtube-transcript-api beautifulsoup4 requests
```

### Run

From project root:

**Option 1 – Crawl the tech library and download all found YouTube transcripts:**

```bash
python scripts/download_youtube_transcripts.py --crawl
```

**Option 2 – Download a specific video by URL or video ID:**

```bash
python scripts/download_youtube_transcripts.py "https://www.youtube.com/watch?v=4fUaFuf3wH0"
python scripts/download_youtube_transcripts.py 4fUaFuf3wH0
```

- **Output:** `data/doodle_labs/<name>-<video_id>.txt` (e.g. `mesh-rider-gui-walkthrough-4fUaFuf3wH0.txt`)
- **No API key** required (uses `youtube-transcript-api`)

### After running

In the RAG chatbot, click **Update with New Documents** to index the new transcript files. Queries like “how to configure channel and bandwidth in Simple Config” will then use the GUI walkthrough transcript when relevant.
