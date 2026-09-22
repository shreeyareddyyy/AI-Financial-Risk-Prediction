import sqlite3
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(BASE_DIR, "fraud_transactions.db")

conn = sqlite3.connect(DB_PATH)

conn.execute("""
CREATE TABLE IF NOT EXISTS fraud_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    amount REAL,
    fraud_prediction INTEGER,
    risk_score REAL,
    risk_level TEXT,
    alert INTEGER,
    feedback TEXT
)
""")

conn.commit()
conn.close()

print("Database initialized successfully!")