"""
Run the full pipeline: pulls from every configured source, analyzes each
item through the same sentiment/risk engine, aggregates by topic, and
writes one combined Excel report.

Fill in CONFIG below with your own credentials, then run:
    python run_pipeline.py          # single run, one-off snapshot
    python run_pipeline.py --watch  # repeats on a schedule (see run_scheduled)

Any source left blank/unconfigured is skipped automatically, so this
runs fine even before you've set up Telegram or YouTube - useful for
testing the rest of the pipeline first.

ON "DOES SENTIMENT UPDATE OVER TIME?"
--------------------------------------
Each call to run() is a snapshot - it pulls whatever is out there RIGHT
NOW and analyzes it once. Nothing updates on its own between runs; there
is no background listener. To get sentiment that tracks changing news/
social activity, you either re-run the script (manually, or on a
schedule via run_scheduled() below / a cron job / Windows Task
Scheduler), or wire run() into the dashboard's refresh button so it
re-pulls on demand. run_scheduled() also keeps a trend log so you can
see how sentiment shifted between runs, not just the latest snapshot.
"""

import csv
import os
import time
from datetime import datetime

from sentiment_pipeline import (
    analyze_text, analyze_image, analyze_telegram_channel,
    analyze_youtube_comments, analyze_youtube_by_keyword,
    analyze_news_feed, analyze_news_by_keyword, DEFAULT_NEWS_FEEDS,
    export_to_excel, aggregate_risk,
)

# ---------------------------------------------------------------------
# CONFIG - fill these in with your own credentials / targets
# ---------------------------------------------------------------------
CONFIG = {
    "telegram": {
        "enabled": False,             # set True once you have credentials
        "api_id": None,               # from https://my.telegram.org
        "api_hash": None,
        # Multiple channels are already supported - just list as many as
        # you want, one per line. These three are established news-outlet
        # channels rather than "signal"/trading-tip channels, which tend
        # to be promotional and add noise rather than organic sentiment:
        #   @cointelegraph   - Cointelegraph, established crypto news outlet
        #   @CoinDeskGlobal  - CoinDesk's official real-time news feed
        #   @WatcherGuru     - fast, general crypto/market breaking-news feed
        # None of these cover gold specifically - gold-focused channels on
        # Telegram skew toward paid trading-signal groups, which are a poor
        # fit for a sentiment project. The news.google.com RSS search in
        # analyze_news_by_keyword() covers gold reliably instead.
        "channels": ["@cointelegraph", "@CoinDeskGlobal", "@WatcherGuru"],
        "limit_per_channel": 30,
    },
    "youtube": {
        "enabled": True,             # set True once you have an API key
        "api_key": "AIzaSyDCfBVyv4TDClRfkJIBTrfj0YdAz8b4kGw",              # from Google Cloud Console
        "mode": "keyword",            # "keyword" (auto-discover videos) or "video_ids" (hand-picked)

        # mode = "keyword": searches YouTube itself and analyzes whatever it finds
        "keywords": ["gold price today", "crypto crash", "stock market fraud india"],
        "max_videos_per_keyword": 5,
        "include_comments": True,
        "comments_per_video": 30,

        # mode = "video_ids": analyze only these specific videos (no search)
        "video_ids": ["dQw4w9WgXcQ"],
        "max_comments": 30,
    },
    "news": {
        "enabled": True,              # no credentials needed
        "mode": "keyword",            # "keyword" (auto-discover) or "feeds" (fixed RSS URLs)

        # mode = "keyword": searches Google News itself for each term
        "keywords": ["gold price", "cryptocurrency crash", "stock market fraud india", "sensex today"],
        "region": "IN",
        "only_relevant": True,        # drop headlines that aren't Gold/Crypto/Market/Fraud tagged

        # mode = "feeds": use fixed RSS feed URLs instead of searching
        "feeds": DEFAULT_NEWS_FEEDS,
    },
    "images": {
        "enabled": True,
        "paths": [],                  # e.g. ["screenshots/post1.png"]
    },
}

TREND_LOG_PATH = "sentiment_trend_log.csv"


def run(quiet: bool = False):
    """One snapshot: pulls from every enabled source, analyzes, aggregates,
    writes an Excel report, and returns (all_results, summary) for
    run_scheduled() to log."""
    all_results = []

    if CONFIG["telegram"]["enabled"]:
        for channel in CONFIG["telegram"]["channels"]:
            if not quiet:
                print(f"Fetching Telegram: {channel}")
            all_results += analyze_telegram_channel(
                channel,
                CONFIG["telegram"]["api_id"],
                CONFIG["telegram"]["api_hash"],
                limit=CONFIG["telegram"]["limit_per_channel"],
            )

    if CONFIG["youtube"]["enabled"]:
        yt = CONFIG["youtube"]
        if yt["mode"] == "keyword":
            for kw in yt["keywords"]:
                if not quiet:
                    print(f"Searching YouTube: \"{kw}\"")
                all_results += analyze_youtube_by_keyword(
                    kw, yt["api_key"], max_videos=yt["max_videos_per_keyword"],
                    include_comments=yt["include_comments"],
                    comments_per_video=yt["comments_per_video"],
                )
        else:  # mode == "video_ids"
            for vid in yt["video_ids"]:
                if not quiet:
                    print(f"Fetching YouTube comments: {vid}")
                all_results += analyze_youtube_comments(vid, yt["api_key"], yt["max_comments"])

    if CONFIG["news"]["enabled"]:
        news = CONFIG["news"]
        if news["mode"] == "keyword":
            if not quiet:
                print(f"Searching news for: {', '.join(news['keywords'])}")
            all_results += analyze_news_by_keyword(
                news["keywords"], region=news["region"], only_relevant=news["only_relevant"]
            )
        else:  # mode == "feeds"
            if not quiet:
                print("Fetching news RSS feeds...")
            all_results += analyze_news_feed(news["feeds"])

    if CONFIG["images"]["enabled"]:
        for path in CONFIG["images"]["paths"]:
            if not quiet:
                print(f"OCR-ing image: {path}")
            all_results.append(analyze_image(path))

    if not all_results:
        if not quiet:
            print("No sources enabled/configured - nothing to analyze. "
                  "Set at least one CONFIG[...]['enabled'] = True and fill in its details.")
        return [], {}

    summary = aggregate_risk(all_results)
    if not quiet:
        print(f"\nTotal items analyzed: {len(all_results)}")
        for topic, s in summary.items():
            print(f"  {topic:<10} count={s['count']:<4} avg_sentiment={s['avg_sentiment']:<7} "
                  f"high_risk={s['high_risk_count']}")

    out_path = export_to_excel(all_results, "combined_sentiment_report.xlsx")
    if not quiet:
        print(f"\nExcel report written to: {out_path}")

    return all_results, summary


def _log_trend(summary: dict):
    """Append this run's per-topic summary to a CSV so sentiment over time
    is visible across runs, not just the latest snapshot."""
    is_new = not os.path.exists(TREND_LOG_PATH)
    with open(TREND_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "topic", "count", "avg_sentiment", "high_risk_count"])
        timestamp = datetime.now().isoformat(timespec="seconds")
        for topic, s in summary.items():
            writer.writerow([timestamp, topic, s["count"], s["avg_sentiment"], s["high_risk_count"]])


def run_scheduled(interval_minutes: int = 30, iterations: int = None):
    """Re-run the pipeline on a fixed schedule so sentiment tracks new
    content instead of staying frozen at one snapshot. Each run's per-topic
    summary is appended to sentiment_trend_log.csv - open that in Excel to
    see how avg_sentiment/high_risk_count moved between runs, which is what
    actually shows a "trend" rather than a single point.

    interval_minutes: gap between runs
    iterations: stop after this many runs (None = run forever, until Ctrl+C)

    For a real deployment, running this as a background/cron job (Linux
    cron, Windows Task Scheduler, or a systemd timer) is more robust than
    leaving a terminal open - but this loop is enough for a demo/viva.
    """
    count = 0
    print(f"Starting scheduled run every {interval_minutes} minute(s). Press Ctrl+C to stop.")
    try:
        while iterations is None or count < iterations:
            count += 1
            print(f"\n=== Run {count} at {datetime.now().isoformat(timespec='seconds')} ===")
            _, summary = run()
            if summary:
                _log_trend(summary)
            if iterations is None or count < iterations:
                time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    import sys
    if "--watch" in sys.argv:
        run_scheduled(interval_minutes=30)
    else:
        run()