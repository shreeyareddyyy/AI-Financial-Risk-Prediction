import sqlite3
import os
from datetime import datetime


# Always store the database in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "fraud_transactions.db"
)


def get_connection():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    """Create the transactions table if it does not already exist."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            amount REAL,
            fraud_prediction INTEGER NOT NULL,
            anomaly_score REAL NOT NULL,
            fraud_risk_score REAL NOT NULL,
            fraud_risk_level TEXT NOT NULL,
            alert INTEGER NOT NULL,
            message TEXT NOT NULL,
            recommended_action TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_transaction(
    user_id,
    amount,
    fraud_prediction,
    anomaly_score,
    fraud_risk_score,
    fraud_risk_level,
    alert,
    message,
    recommended_action
):
    """Save one scored transaction."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO transactions (
            user_id,
            timestamp,
            amount,
            fraud_prediction,
            anomaly_score,
            fraud_risk_score,
            fraud_risk_level,
            alert,
            message,
            recommended_action
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(user_id),
        datetime.now().isoformat(timespec="seconds"),
        amount,
        int(fraud_prediction),
        float(anomaly_score),
        float(fraud_risk_score),
        str(fraud_risk_level),
        int(alert),
        str(message),
        str(recommended_action)
    ))

    connection.commit()
    connection.close()


def get_transactions(user_id=None):
    """
    Return transaction history.

    If user_id is provided, only that user's transactions are returned.
    Otherwise, all transactions are returned.
    """

    connection = get_connection()
    cursor = connection.cursor()

    if user_id is not None:

        cursor.execute("""
            SELECT
                id,
                user_id,
                timestamp,
                amount,
                fraud_prediction,
                anomaly_score,
                fraud_risk_score,
                fraud_risk_level,
                alert,
                message,
                recommended_action
            FROM transactions
            WHERE user_id = ?
            ORDER BY id DESC
        """, (str(user_id),))

    else:

        cursor.execute("""
            SELECT
                id,
                user_id,
                timestamp,
                amount,
                fraud_prediction,
                anomaly_score,
                fraud_risk_score,
                fraud_risk_level,
                alert,
                message,
                recommended_action
            FROM transactions
            ORDER BY id DESC
        """)

    rows = cursor.fetchall()

    connection.close()

    return rows