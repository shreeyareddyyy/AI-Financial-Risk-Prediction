import sqlite3
import os
from datetime import datetime


# Always store the database in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "fraud_transactions.db"
)


def init_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            fraud_prediction INTEGER,
            anomaly_score REAL,
            fraud_risk_score REAL,
            fraud_risk_level TEXT,
            alert INTEGER,
            message TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_transaction(
    amount,
    fraud_prediction,
    anomaly_score,
    fraud_risk_score,
    fraud_risk_level,
    alert,
    message
):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO transactions (
            timestamp,
            amount,
            fraud_prediction,
            anomaly_score,
            fraud_risk_score,
            fraud_risk_level,
            alert,
            message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        amount,
        fraud_prediction,
        anomaly_score,
        fraud_risk_score,
        fraud_risk_level,
        int(alert),
        message
    ))

    connection.commit()
    connection.close()


def get_transactions():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            amount,
            fraud_prediction,
            anomaly_score,
            fraud_risk_score,
            fraud_risk_level,
            alert,
            message
        FROM transactions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows