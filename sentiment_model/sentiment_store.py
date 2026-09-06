"""
sentiment_store.py

The shared live store for the Sentiment module, per the Phase 2 roadmap's
team-wide decision: "A single shared store (SQLite is enough) that every
module writes its latest results into ... the dashboard reads that store
continuously instead of a static file."

This file owns the SQLite schema and every read/write helper for
sentiment data specifically. Raghavi's fraud module and Varshini's
price module write their own tables into the SAME .db file (agree on
one path, e.g. shared_store.db at the repo root) - they don't need to
touch this file, they just open the same sqlite3 connection and create
their own tables. Sanjanna's dashboard imports the `get_*` functions
below to read sentiment data live.

WHY SQLITE AND NOT EXCEL
-------------------------
Phase 2 correction #8: "Remove Excel as the source of truth ... Excel
becomes an export option only, never the thing the dashboard reads
from." SQLite supports concurrent reads while a write is happening
(WAL mode, enabled below), needs no server, and is a single file the
whole team can share via the repo. Excel export still exists
(see run_pipeline.py: export_snapshot_to_excel) purely for submitting
a static snapshot alongside the report.

SCHEMA
------
news_items         one row per unique ingested item (news/youtube/telegram/...)
sentiment_scores   one row per (news_item, asset) - this is what makes
                    sentiment "per-asset" instead of one blended number
sentiment_trend    periodic rolling snapshots per asset, so the dashboard
                    can show "now / 1h ago / 6h ago / 24h ago"
alerts             sentiment-swing alerts fired by run_pipeline.py
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

DEFAULT_DB_PATH = os.environ.get("SHARED_STORE_PATH", "shared_store.db")

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,      -- news | youtube_video | youtube | telegram | image | audio | video
    source_ref      TEXT,               -- filename / video id / channel#msgid
    url             TEXT,               -- article/video URL if we have one (used for dedup)
    headline        TEXT,               -- short display text (used for near-dup detection)
    raw_text        TEXT,
    published_at    TEXT,               -- from the source feed, if available
    fetched_at      TEXT NOT NULL,      -- when WE first saw it
    UNIQUE(url)
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    news_item_id        INTEGER NOT NULL REFERENCES news_items(id),
    asset               TEXT NOT NULL,      -- Gold | Crypto | Market | Fraud | General
    vader_label         TEXT,
    vader_score         REAL,
    finbert_label       TEXT,
    finbert_score       REAL,
    finbert_confidence  REAL,
    final_label         TEXT NOT NULL,      -- FinBERT is primary; this is what the dashboard shows
    final_score         REAL NOT NULL,
    risk_level          TEXT NOT NULL,
    risk_score          INTEGER NOT NULL,
    scored_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment_trend (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    snapshot_at     TEXT NOT NULL,
    avg_sentiment   REAL NOT NULL,
    item_count      INTEGER NOT NULL,
    high_risk_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT NOT NULL,
    alert_type  TEXT NOT NULL,      -- sentiment_swing
    message     TEXT NOT NULL,
    old_value   REAL,
    new_value   REAL,
    delta       REAL,
    severity    TEXT NOT NULL,      -- Medium | High
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_asset ON sentiment_scores(asset);
CREATE INDEX IF NOT EXISTS idx_scores_item ON sentiment_scores(news_item_id);
CREATE INDEX IF NOT EXISTS idx_trend_asset_time ON sentiment_trend(asset, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_items_fetched ON news_items(fetched_at);
"""


@contextmanager
def get_conn(db_path: str = DEFAULT_DB_PATH):
    """Every read/write goes through this so WAL mode (safe concurrent
    reads while the pipeline is writing) is always on, and connections
    are always closed."""
    with _lock:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------
# WRITES
# ---------------------------------------------------------------------

def url_seen(url: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    if not url:
        return False
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT 1 FROM news_items WHERE url = ?", (url,)).fetchone()
        return row is not None


def recent_headlines(hours: int = 24, db_path: str = DEFAULT_DB_PATH) -> list:
    """Headlines from the last N hours, used for near-duplicate checking
    (same story, different outlet -> different URL, near-identical headline)."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT headline FROM news_items WHERE fetched_at >= ? AND headline IS NOT NULL",
            (cutoff,),
        ).fetchall()
    return [r["headline"] for r in rows]


def insert_news_item(source_type, source_ref, url, headline, raw_text,
                      published_at=None, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert one ingested item. Returns its new id. If `url` collides
    with an existing row (UNIQUE constraint), returns the EXISTING id
    instead of inserting a duplicate - belt-and-suspenders alongside the
    url_seen()/near-duplicate checks the caller should already be doing."""
    fetched_at = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        try:
            cur = conn.execute(
                "INSERT INTO news_items (source_type, source_ref, url, headline, "
                "raw_text, published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source_type, source_ref, url, headline, raw_text, published_at, fetched_at),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            row = conn.execute("SELECT id FROM news_items WHERE url = ?", (url,)).fetchone()
            return row["id"]


def insert_sentiment_score(news_item_id, asset, vader_label, vader_score,
                            finbert_label, finbert_score, finbert_confidence,
                            final_label, final_score, risk_level, risk_score,
                            db_path: str = DEFAULT_DB_PATH):
    scored_at = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO sentiment_scores (news_item_id, asset, vader_label, vader_score, "
            "finbert_label, finbert_score, finbert_confidence, final_label, final_score, "
            "risk_level, risk_score, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (news_item_id, asset, vader_label, vader_score, finbert_label, finbert_score,
             finbert_confidence, final_label, final_score, risk_level, risk_score, scored_at),
        )


def insert_trend_snapshot(asset, avg_sentiment, item_count, high_risk_count,
                           db_path: str = DEFAULT_DB_PATH):
    snapshot_at = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO sentiment_trend (asset, snapshot_at, avg_sentiment, item_count, "
            "high_risk_count) VALUES (?,?,?,?,?)",
            (asset, snapshot_at, avg_sentiment, item_count, high_risk_count),
        )


def insert_alert(asset, alert_type, message, old_value, new_value, delta, severity,
                  db_path: str = DEFAULT_DB_PATH):
    created_at = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO alerts (asset, alert_type, message, old_value, new_value, delta, "
            "severity, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (asset, alert_type, message, old_value, new_value, delta, severity, created_at),
        )


# ---------------------------------------------------------------------
# READS - what Sanjanna's dashboard (or evaluate.py / a notebook) calls
# ---------------------------------------------------------------------

def get_latest_news(limit: int = 50, asset: str = None, db_path: str = DEFAULT_DB_PATH) -> list:
    """Latest scored items, newest first - the "Latest news" feed with
    relative timestamps the dashboard renders. One row per (item, asset)
    so a mixed-topic article appears once per asset it was scored for."""
    query = """
        SELECT n.id, n.source_type, n.source_ref, n.url, n.headline, n.raw_text,
               n.fetched_at, s.asset, s.final_label, s.final_score,
               s.finbert_confidence, s.risk_level, s.risk_score
        FROM sentiment_scores s
        JOIN news_items n ON n.id = s.news_item_id
    """
    params = []
    if asset:
        query += " WHERE s.asset = ?"
        params.append(asset)
    query += " ORDER BY n.fetched_at DESC LIMIT ?"
    params.append(limit)
    with get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_asset_summary(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Current (all-time-so-far) per-asset rollup - count, avg sentiment,
    high-risk count. What feeds the unified risk score's sentiment input."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT asset, COUNT(*) AS count, AVG(final_score) AS avg_sentiment,
                      SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_count
               FROM sentiment_scores GROUP BY asset"""
        ).fetchall()
    return {r["asset"]: dict(r) for r in rows}


def get_trend(asset: str, hours: int = 24, db_path: str = DEFAULT_DB_PATH) -> list:
    """Trend points for one asset over the last N hours, oldest first -
    what the sentiment-vs-price overlay chart and the "now/1h/6h/24h ago"
    view plot."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT snapshot_at, avg_sentiment, item_count, high_risk_count "
            "FROM sentiment_trend WHERE asset = ? AND snapshot_at >= ? ORDER BY snapshot_at ASC",
            (asset, cutoff),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_trend_value(asset: str, db_path: str = DEFAULT_DB_PATH):
    """Most recent trend snapshot for one asset, or None if there isn't one
    yet. Used by the swing-alert check to compare "now" against "previous run"."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT snapshot_at, avg_sentiment FROM sentiment_trend "
            "WHERE asset = ? ORDER BY snapshot_at DESC LIMIT 1",
            (asset,),
        ).fetchone()
    return dict(row) if row else None


def get_news_impact(limit: int = 20, min_abs_score: float = 0.4,
                     db_path: str = DEFAULT_DB_PATH) -> list:
    """Phase 2 correction #6, "News impact view": the subset of scored
    items worth surfacing as a structured signal rather than a raw
    headline - strong sentiment (|score| >= min_abs_score) or elevated
    risk. Each row already carries sentiment, confidence, asset, and
    risk impact (final_label, finbert_confidence, asset, risk_level/
    risk_score). A price-impact signal column is intentionally NOT
    included here: that needs Varshini's live price/volatility output
    to mean anything, so it belongs where sentiment and price are
    combined (Sanjanna's unified risk view), not fabricated here."""
    query = """
        SELECT n.id, n.headline, n.raw_text, n.url, n.fetched_at, n.source_type,
               s.asset, s.final_label, s.final_score, s.finbert_confidence,
               s.vader_label, s.vader_score, s.risk_level, s.risk_score
        FROM sentiment_scores s
        JOIN news_items n ON n.id = s.news_item_id
        WHERE ABS(s.final_score) >= ? OR s.risk_level IN ('Medium', 'High')
        ORDER BY s.risk_score DESC, ABS(s.final_score) DESC, n.fetched_at DESC
        LIMIT ?
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(query, (min_abs_score, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent_alerts(limit: int = 20, db_path: str = DEFAULT_DB_PATH) -> list:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialized shared store at: {os.path.abspath(DEFAULT_DB_PATH)}")
