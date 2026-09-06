"""
sentiment_engine.py  (Phase 2 rewrite of sentiment_pipeline.py)

Owner: Shreeya (Sentiment, Risk Score & Dashboard feed)

WHAT CHANGED FROM PHASE 1 AND WHY
----------------------------------
Phase 1's sentiment_pipeline.py was solid (VADER + a finance keyword
lexicon), but the Phase 2 review flagged three specific gaps for this
module, all fixed here:

  1. "Add FinBERT as the primary financial sentiment classifier, with
     VADER kept as a supporting/comparison signal."
     -> _score_finbert() is now the PRIMARY score (final_label/
     final_score). VADER still runs on every item and is stored
     alongside it (vader_label/vader_score) so the two can be compared
     in the report, exactly like transformer_score.py + evaluate.py
     already let you do for the labeled sample.

  2. "Score sentiment separately per asset ... since one article can be
     positive for gold and negative for crypto at once. Confirm asset
     tagging happens before scoring."
     -> analyze_text_multi_asset() tags topics PER SENTENCE first,
     groups sentences by the asset they're actually about, and only
     THEN runs FinBERT/VADER on each asset's own sentences. A single
     mixed article now produces one row per asset it actually discusses,
     each with its own independent sentiment - not one blended number
     copy-pasted across every topic tag it happened to match (that was
     the Phase 1 behaviour, and it's what the review called out).

  3. "Automatic news collection ... duplicate removal ... Excel becomes
     an export option only, never the thing the dashboard reads from."
     -> process_item() is the new single entry point every adapter
     funnels through: it checks the shared store for an exact URL match
     AND a near-duplicate headline (same story, different outlet) BEFORE
     paying for a FinBERT call, then writes straight into
     sentiment_store.py's SQLite tables. run_pipeline.py reads this
     module; the dashboard reads the store, never this module's return
     values directly.

Everything else - the four source adapters (news/YouTube/Telegram/
image/audio/video), the finance risk-keyword lexicon, the risk
scoring logic - is carried over from Phase 1 with only the changes
needed to plug into the new per-asset + shared-store flow.
"""

import os
import re
import difflib
from dataclasses import dataclass, field
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sentiment_store as store

# ---------------------------------------------------------------------
# 1. FINANCE-DOMAIN LEXICON  (unchanged from Phase 1)
# ---------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "Fraud": [
        "scam", "rug pull", "ponzi", "fraud", "fraudulent", "phishing",
        "hack", "hacked", "exploit", "ransom", "laundering", "fake",
        "counterfeit", "insider trading", "manipulation", "pump and dump",
    ],
    "Crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "altcoin", "token", "blockchain", "wallet", "exchange", "defi",
        "nft", "mining", "halving", "stablecoin",
    ],
    "Gold": [
        "gold", "bullion", "xau", "precious metal", "gold price",
        "gold reserve", "gold etf", "sovereign gold bond",
    ],
    "Market": [
        "stock market", "nifty", "sensex", "nasdaq", "dow jones",
        "s&p", "shares", "equity", "ipo", "bull market", "bear market",
        "recession", "inflation", "interest rate", "fed", "rbi",
    ],
}

RISK_BOOST_WORDS = {
    "crash": (-0.6, 3), "plunge": (-0.5, 3), "collapse": (-0.6, 3),
    "scam": (-0.7, 3), "rug pull": (-0.8, 3), "hacked": (-0.6, 3),
    "delisted": (-0.5, 2), "banned": (-0.4, 2), "frozen": (-0.4, 2),
    "investigation": (-0.3, 2), "probe": (-0.3, 2), "lawsuit": (-0.3, 2),
    "surge": (0.5, 1), "rally": (0.5, 1), "all-time high": (0.6, 1),
    "record high": (0.6, 1), "breakout": (0.4, 1),
}

# "Positive for X, negative for Y in the same article" is the whole
# point of per-asset scoring (module docstring) - but that split often
# happens WITHIN one grammatical sentence via a connector ("...hack,
# while gold hit a record high..."), not just at a full stop. So text
# is segmented in two passes: sentence punctuation first, then clause
# connectors within each sentence.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(r"\s*,?\s+\b(?:while|but|whereas|meanwhile|however)\b\s*", re.IGNORECASE)

# How similar two headlines need to be (0-1, difflib ratio) to count as
# the same story from a different outlet. 0.85 tolerates small wording
# differences ("Gold hits record high" vs "Gold prices hit all-time high")
# while still catching genuinely different headlines.
DUPLICATE_HEADLINE_THRESHOLD = 0.85


@dataclass
class AssetSentimentResult:
    """One row = one (news item, asset) pair. A single article about both
    Bitcoin and Gold produces two of these, each independently scored."""
    source_type: str
    source_ref: str
    asset: str                      # Gold | Crypto | Market | Fraud | General
    text_used: str                  # the sentence(s) actually scored for this asset
    vader_label: str
    vader_score: float
    finbert_label: str
    finbert_score: float            # signed: +confidence / -confidence / 0
    finbert_confidence: float       # 0-1, the model's confidence in finbert_label
    final_label: str                # = finbert_label (FinBERT is primary)
    final_score: float              # = finbert_score
    risk_level: str
    risk_score: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ---------------------------------------------------------------------
# 2. VADER  (secondary / comparison signal - see transformer_score.py
#    and evaluate.py for the accuracy comparison this claim is based on)
# ---------------------------------------------------------------------

_vader = SentimentIntensityAnalyzer()


def _score_vader(text: str):
    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(compound, 3)


# ---------------------------------------------------------------------
# 3. FinBERT  (primary classifier)
#    ProsusAI/finbert - trained on financial text (Reuters headlines +
#    analyst reports), not general Twitter chatter, so it reads finance
#    phrasing ("guidance cut", "beat estimates") the way VADER's generic
#    lexicon can't. This is what run_pipeline.py's docstring / the report
#    should cite when asked "why this model".
# ---------------------------------------------------------------------

_finbert_pipeline = None
FINBERT_MODEL_NAME = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")


def _get_finbert():
    """Lazy-loaded so importing this module (or running evaluate.py,
    tune_thresholds.py, etc.) never pays the model-load cost unless a
    FinBERT score is actually requested."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers torch"
            )
        _finbert_pipeline = pipeline(
            "sentiment-analysis", model=FINBERT_MODEL_NAME, truncation=True, max_length=512,
        )
    return _finbert_pipeline


def _score_finbert_batch(texts: list) -> list:
    """Score many texts in one batched call - used so a single article
    with 3 asset-groups (or a run of 50 headlines) doesn't cost 3-50
    separate model calls."""
    if not texts:
        return []
    classifier = _get_finbert()
    safe_texts = [t if t and t.strip() else "." for t in texts]
    raw = classifier(safe_texts)
    out = []
    for r in raw:
        label = r["label"].capitalize()
        confidence = round(float(r["score"]), 3)
        if label == "Positive":
            signed = confidence
        elif label == "Negative":
            signed = -confidence
        else:
            signed = 0.0
        out.append((label, round(signed, 3), confidence))
    return out


# ---------------------------------------------------------------------
# 4. TOPIC DETECTION + PER-SENTENCE ASSET TAGGING (before any scoring)
# ---------------------------------------------------------------------

def _detect_topics(text_lower: str) -> list:
    """Returns every topic whose keywords appear in this piece of text.
    No 'General' fallback here on purpose - that's decided once, at the
    whole-article level, only if NOTHING tagged anywhere (see
    analyze_text_multi_asset)."""
    return [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in text_lower for w in words)]


def _split_sentences(text: str) -> list:
    """Sentence split, then clause split within each sentence, so a
    single comma-joined sentence spanning two assets ("X crashed, while
    Y hit a high") still yields two separately-taggable segments."""
    text = text.strip()
    if not text:
        return []
    segments = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        segments.extend(c.strip() for c in CLAUSE_SPLIT_RE.split(sentence) if c.strip())
    return segments


def _risk_for_text(text_lower: str, final_score: float, asset: str):
    """Same risk-scoring logic as Phase 1, scoped to one asset's own
    text instead of the whole article, and using FinBERT's score
    (the new primary signal) for the sentiment-driven component."""
    adj, weight = 0, 0
    for phrase, (_, risk_w) in RISK_BOOST_WORDS.items():
        if phrase in text_lower:
            weight += risk_w
    risk_score = weight
    if final_score <= -0.6:
        risk_score += 2
    elif final_score <= -0.2:
        risk_score += 1
    if asset == "Fraud":
        risk_score += 2

    if risk_score >= 5:
        risk_level = "High"
    elif risk_score >= 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    return risk_level, risk_score


# ---------------------------------------------------------------------
# 5. CORE: asset tagging BEFORE scoring, then independent per-asset scoring
# ---------------------------------------------------------------------

def analyze_text_multi_asset(text: str, source_type: str = "text",
                              source_ref: str = None) -> list:
    """The core function every adapter and the swap-in-a-live-API path
    both funnel into. Returns a list of AssetSentimentResult - one per
    asset the text actually discusses (usually 1, sometimes 2-3 for a
    roundup article, "General" if nothing finance-specific matched)."""
    text = (text or "").strip()
    source_ref = source_ref or (text[:40] + "..." if len(text) > 40 else text)
    if not text:
        return []

    sentences = _split_sentences(text)

    # Step 1: tag every sentence with the assets IT discusses. This is
    # the "asset tagging happens before scoring" requirement - we decide
    # what belongs to Gold vs Crypto vs Fraud before any model sees it.
    asset_sentences = {}  # asset -> list[str]
    for sentence in sentences:
        topics = _detect_topics(sentence.lower())
        for topic in topics:
            asset_sentences.setdefault(topic, []).append(sentence)

    if not asset_sentences:
        # Nothing in the whole article matched a specific asset - score
        # it once as General rather than manufacturing false asset tags.
        asset_sentences["General"] = sentences or [text]

    assets = list(asset_sentences.keys())
    asset_texts = [" ".join(asset_sentences[a]) for a in assets]

    # Step 2: score each asset's own text independently. VADER first
    # (cheap, local), FinBERT batched across all assets in this one item.
    finbert_results = _score_finbert_batch(asset_texts)

    results = []
    for asset, asset_text, (fb_label, fb_score, fb_conf) in zip(assets, asset_texts, finbert_results):
        v_label, v_score = _score_vader(asset_text)
        risk_level, risk_score = _risk_for_text(asset_text.lower(), fb_score, asset)

        results.append(AssetSentimentResult(
            source_type=source_type,
            source_ref=source_ref,
            asset=asset,
            text_used=asset_text,
            vader_label=v_label,
            vader_score=v_score,
            finbert_label=fb_label,
            finbert_score=fb_score,
            finbert_confidence=fb_conf,
            final_label=fb_label,
            final_score=fb_score,
            risk_level=risk_level,
            risk_score=risk_score,
        ))
    return results


# ---------------------------------------------------------------------
# 6. DEDUPLICATION  (Phase 2 correction: "implement URL deduplication
#    plus headline-similarity checks")
# ---------------------------------------------------------------------

def _normalize_headline(headline: str) -> str:
    h = re.sub(r"[^a-z0-9\s]", " ", (headline or "").lower())
    return " ".join(h.split())


def is_duplicate(headline: str, url: str = None, db_path: str = store.DEFAULT_DB_PATH) -> bool:
    """True if this URL has already been ingested, or if a
    near-identical headline was ingested in the last 24h (same story,
    different outlet)."""
    if url and store.url_seen(url, db_path):
        return True
    if not headline:
        return False
    norm = _normalize_headline(headline)
    if not norm:
        return False
    for existing in store.recent_headlines(hours=24, db_path=db_path):
        if difflib.SequenceMatcher(None, norm, _normalize_headline(existing)).ratio() >= DUPLICATE_HEADLINE_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------
# 7. INGESTION ENTRY POINT  (dedup -> score -> write to shared store)
# ---------------------------------------------------------------------

def process_item(text: str, source_type: str, source_ref: str = None,
                  url: str = None, headline: str = None, published_at: str = None,
                  db_path: str = store.DEFAULT_DB_PATH) -> list:
    """News APIs/RSS -> sentiment engine -> shared database, per the
    Phase 2 target flow. Returns [] (and does NOT call FinBERT) if this
    item is a duplicate - dedup runs before the expensive model call,
    not after."""
    headline = headline or (text[:80] if text else source_ref)
    if is_duplicate(headline, url, db_path):
        return []

    results = analyze_text_multi_asset(text, source_type=source_type, source_ref=source_ref)
    if not results:
        return []

    item_id = store.insert_news_item(
        source_type=source_type, source_ref=source_ref, url=url,
        headline=headline, raw_text=text, published_at=published_at, db_path=db_path,
    )
    for r in results:
        store.insert_sentiment_score(
            news_item_id=item_id, asset=r.asset,
            vader_label=r.vader_label, vader_score=r.vader_score,
            finbert_label=r.finbert_label, finbert_score=r.finbert_score,
            finbert_confidence=r.finbert_confidence,
            final_label=r.final_label, final_score=r.final_score,
            risk_level=r.risk_level, risk_score=r.risk_score, db_path=db_path,
        )
    return results


# ---------------------------------------------------------------------
# 8. SOURCE ADAPTERS - extract text (+ URL when available), then
#    funnel into process_item(). Same four sources as Phase 1.
# ---------------------------------------------------------------------

def analyze_image(path: str, db_path: str = store.DEFAULT_DB_PATH) -> list:
    import pytesseract
    from PIL import Image

    text = pytesseract.image_to_string(Image.open(path))
    return process_item(text, source_type="image", source_ref=os.path.basename(path), db_path=db_path)


def analyze_audio(path: str, db_path: str = store.DEFAULT_DB_PATH) -> list:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition service unavailable: {e}")

    return process_item(text, source_type="audio", source_ref=os.path.basename(path), db_path=db_path)


def analyze_video(path: str, db_path: str = store.DEFAULT_DB_PATH) -> list:
    from pydub import AudioSegment
    import tempfile
    import speech_recognition as sr

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "extracted.wav")
        audio = AudioSegment.from_file(path)
        audio.export(wav_path, format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            raise RuntimeError(f"Speech recognition service unavailable: {e}")

    return process_item(text, source_type="video", source_ref=os.path.basename(path), db_path=db_path)


def analyze_news_feed(feed_urls: list, db_path: str = store.DEFAULT_DB_PATH) -> list:
    """Pull headlines from RSS feeds and analyze each NEW one. Dedup
    (URL + near-identical headline) happens inside process_item(), so
    re-running this against a feed you already pulled 10 minutes ago
    only analyzes what's actually new."""
    import feedparser

    all_results = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            headline = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link")
            published = entry.get("published")
            combined = f"{headline}. {summary}"
            all_results += process_item(
                combined, source_type="news", source_ref=headline[:60],
                url=link, headline=headline, published_at=published, db_path=db_path,
            )
    return all_results


def build_google_news_rss_url(query: str, region: str = "IN", lang: str = "en") -> str:
    import urllib.parse
    encoded = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"


def analyze_news_by_keyword(keywords: list, region: str = "IN", lang: str = "en",
                             only_relevant: bool = True, db_path: str = store.DEFAULT_DB_PATH) -> list:
    all_results = []
    for kw in keywords:
        url = build_google_news_rss_url(kw, region, lang)
        all_results += analyze_news_feed([url], db_path=db_path)

    if only_relevant:
        all_results = [r for r in all_results if r.asset != "General"]
    return all_results


DEFAULT_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=gold+price+OR+cryptocurrency+OR+stock+market+fraud&hl=en-IN&gl=IN&ceid=IN:en",
    "https://finance.yahoo.com/news/rssindex",
]


def analyze_telegram_channel(channel_username: str, api_id: int, api_hash: str,
                              limit: int = 50, db_path: str = store.DEFAULT_DB_PATH) -> list:
    from telethon.sync import TelegramClient

    all_results = []
    with TelegramClient("sentiment_session", api_id, api_hash) as client:
        for message in client.iter_messages(channel_username, limit=limit):
            if message.text:
                pseudo_url = f"telegram://{channel_username}/{message.id}"
                all_results += process_item(
                    message.text, source_type="telegram",
                    source_ref=f"{channel_username}#{message.id}",
                    url=pseudo_url, db_path=db_path,
                )
    return all_results


def analyze_youtube_comments(video_id: str, api_key: str, max_results: int = 50,
                              db_path: str = store.DEFAULT_DB_PATH) -> list:
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
    )
    response = request.execute()

    all_results = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        pseudo_url = f"youtube://{video_id}/comment/{item['id']}"
        all_results += process_item(
            comment, source_type="youtube", source_ref=f"{video_id}#{item['id']}",
            url=pseudo_url, db_path=db_path,
        )
    return all_results


def search_youtube_videos(query: str, api_key: str, max_results: int = 10,
                           order: str = "relevance", published_after: str = None) -> list:
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=api_key)
    kwargs = dict(q=query, part="snippet", type="video", maxResults=max_results, order=order)
    if published_after:
        kwargs["publishedAfter"] = published_after
    response = youtube.search().list(**kwargs).execute()

    videos = []
    for item in response.get("items", []):
        snippet = item["snippet"]
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": snippet["title"],
            "description": snippet["description"],
            "channel": snippet["channelTitle"],
            "published_at": snippet["publishedAt"],
        })
    return videos


def analyze_youtube_by_keyword(query: str, api_key: str, max_videos: int = 5,
                                include_comments: bool = True, comments_per_video: int = 20,
                                published_after: str = None, db_path: str = store.DEFAULT_DB_PATH) -> list:
    all_results = []
    videos = search_youtube_videos(query, api_key, max_results=max_videos, published_after=published_after)

    for v in videos:
        combined = f"{v['title']}. {v['description']}"
        video_url = f"https://www.youtube.com/watch?v={v['video_id']}"
        all_results += process_item(
            combined, source_type="youtube_video",
            source_ref=f"{v['video_id']} ({v['channel']})",
            url=video_url, headline=v["title"], published_at=v["published_at"], db_path=db_path,
        )
        if include_comments:
            try:
                all_results += analyze_youtube_comments(v["video_id"], api_key, max_results=comments_per_video, db_path=db_path)
            except Exception as e:
                print(f"  (comments unavailable for {v['video_id']}: {e})")

    return all_results


# ---------------------------------------------------------------------
# 9. DEMO / SELF-TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":
    store.init_db()
    sample_items = [
        ("Bitcoin crashes 18% after major exchange hack, investigators suspect insider fraud, "
         "while gold prices hit an all-time high as investors flee to safety."),
        "Sensex rallies 500 points as inflation data comes in better than expected.",
        "SEC opens investigation into crypto exchange over alleged rug pull scheme.",
        "Just had a great cup of coffee this morning, nothing special today.",
    ]

    for i, text in enumerate(sample_items):
        results = process_item(text, source_type="demo", source_ref=f"demo-{i}", url=f"demo://{i}")
        for r in results:
            print(f"[{r.asset:<8}] {r.final_label:<9} score={r.final_score:+.3f}  "
                  f"(vader={r.vader_label} {r.vader_score:+.3f})  risk={r.risk_level}  :: {r.text_used[:70]}")
