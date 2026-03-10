"""
Download YouTube video transcripts from Doodle Labs tech library and save for RAG.

- Discovers YouTube links by crawling https://techlibrary.doodlelabs.com/doodle-labs-technical-library
- Fetches captions/transcripts via youtube-transcript-api (no API key)
- Saves each transcript as .txt in data/doodle_labs/ so the chatbot can index them

Usage:
    # Crawl tech library, find all YouTube links, download their transcripts
    python scripts/download_youtube_transcripts.py --crawl

    # Download specific video(s) by URL or video ID
    python scripts/download_youtube_transcripts.py "https://www.youtube.com/watch?v=4fUaFuf3wH0"
    python scripts/download_youtube_transcripts.py 4fUaFuf3wH0

Output:
    data/doodle_labs/mesh-rider-gui-walkthrough-4fUaFuf3wH0.txt  (example)
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "doodle_labs"
BASE_URL = "https://techlibrary.doodlelabs.com"
START_URL = "https://techlibrary.doodlelabs.com/doodle-labs-technical-library"
REQUEST_DELAY = 1.5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (compatible; RAG-doc-ingest/1.0)"
TIMEOUT = 30
CRAWL_PAGE_LIMIT = 80  # max pages to scan for YouTube links

# Known Doodle Labs video IDs -> friendly name for filename (better RAG retrieval)
KNOWN_VIDEOS = {
    "4fUaFuf3wH0": "mesh-rider-gui-walkthrough",
}


def extract_video_id(url_or_id: str) -> str | None:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    s = (url_or_id or "").strip()
    if not s:
        return None
    # Already looks like a video ID (11 chars, alphanumeric + - _)
    if re.match(r"^[\w-]{11}$", s):
        return s
    parsed = urlparse(s)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        if parsed.netloc == "youtu.be":
            return parsed.path.strip("/")[:11] or None
        q = parse_qs(parsed.query)
        vid = (q.get("v") or [None])[0]
        return (vid or "")[:11] or None
    return None


def get_youtube_ids_from_html(html: str, current_url: str) -> set[str]:
    """Extract all YouTube video IDs from a page (links and iframes)."""
    soup = BeautifulSoup(html, "html.parser")
    ids = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "youtube.com" in href or "youtu.be" in href:
            vid = extract_video_id(href)
            if vid:
                ids.add(vid)
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"].strip()
        if "youtube.com" in src or "youtu.be" in src:
            vid = extract_video_id(src)
            if vid:
                ids.add(vid)
    return ids


def get_same_domain_links(html: str) -> list[str]:
    """Get same-domain links from page for crawling."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("#")[0]
        if not href:
            continue
        full = urljoin(BASE_URL + "/", href)
        if "techlibrary.doodlelabs.com" not in full or full.startswith("mailto:"):
            continue
        if re.search(r"\.(jpg|jpeg|png|gif|css|js|zip|pdf)\b", full, re.I):
            continue
        out.append(full)
    return list(dict.fromkeys(out))


def fetch_page(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def crawl_for_youtube_ids(max_pages: int = CRAWL_PAGE_LIMIT) -> set[str]:
    """Crawl tech library from START_URL and collect all YouTube video IDs."""
    to_visit = [START_URL]
    visited = set()
    all_ids = set()
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"  [{len(visited)}] {url[:70]}...")
        html = fetch_page(url)
        time.sleep(REQUEST_DELAY)
        if not html:
            continue
        ids = get_youtube_ids_from_html(html, url)
        all_ids |= ids
        for link in get_same_domain_links(html):
            if link not in visited and link not in to_visit:
                to_visit.append(link)
    return all_ids


def fetch_transcript(video_id: str) -> str | None:
    """Fetch transcript for one video. Returns plain text or None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("  Install: pip install youtube-transcript-api")
        return None
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        if not fetched:
            return None
        # FetchedTranscript is iterable of FetchedTranscriptSnippet with .text
        lines = [s.text.strip() for s in fetched if getattr(s, "text", None)]
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"  Transcript failed {video_id}: {e}")
        return None


def video_id_to_filename(video_id: str, title_hint: str = "") -> str:
    """Safe filename for transcript (RAG will index by filename)."""
    hint = KNOWN_VIDEOS.get(video_id) or title_hint or video_id
    safe = re.sub(r"[^\w-]", "_", str(hint).strip())[:80]
    return f"{safe or 'youtube'}-{video_id}.txt"


def download_transcripts(video_ids: set[str] | list[str]) -> int:
    """Fetch transcript for each ID and save to OUTPUT_DIR. Returns count saved."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for vid in sorted(video_ids):
        text = fetch_transcript(vid)
        time.sleep(0.5)
        if not text or len(text) < 50:
            print(f"  Skip {vid} (no transcript or too short)")
            continue
        fname = video_id_to_filename(vid, f"youtube-{vid}")
        out_path = OUTPUT_DIR / fname
        header = f"Source: https://www.youtube.com/watch?v={vid}\nDoodle Labs Technical Library (video transcript)\n\n"
        out_path.write_text(header + text, encoding="utf-8")
        saved += 1
        print(f"  Saved {fname} ({len(text)} chars)")
    return saved


def main():
    video_ids = set()
    if "--crawl" in sys.argv:
        print("Crawling tech library for YouTube links...")
        video_ids = crawl_for_youtube_ids()
        print(f"Found {len(video_ids)} unique video(s).")
    for arg in sys.argv[1:]:
        if arg == "--crawl":
            continue
        vid = extract_video_id(arg)
        if vid:
            video_ids.add(vid)
    if not video_ids:
        print("Usage:")
        print("  python scripts/download_youtube_transcripts.py --crawl")
        print("  python scripts/download_youtube_transcripts.py <YouTube URL or video ID>")
        print()
        print("Example (Mesh Rider GUI walkthrough):")
        print("  python scripts/download_youtube_transcripts.py https://www.youtube.com/watch?v=4fUaFuf3wH0")
        return 0
    print(f"Downloading transcripts for {len(video_ids)} video(s) -> {OUTPUT_DIR}")
    saved = download_transcripts(video_ids)
    print()
    print(f"Done. Saved {saved} transcript(s). Run 'Update with New Documents' in the chatbot to index them.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
