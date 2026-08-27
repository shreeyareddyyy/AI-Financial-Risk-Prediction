import sqlite3
import os
from datetime import datetime


# Always store the database in the project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "fraud_transactions.db"
)


def init_database():
    """Create all required database tables."""

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Main transaction history table
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
            message TEXT,
            user_id TEXT,
            recommended_action TEXT
        )
    """)

    # Feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            timestamp TEXT NOT NULL
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
    message,
    user_id=None,
    recommended_action=None
):
    """Save a fraud detection result."""

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
            message,
            user_id,
            recommended_action
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        amount,
        fraud_prediction,
        anomaly_score,
        fraud_risk_score,
        fraud_risk_level,
        int(alert),
        message,
        user_id,
        recommended_action
    ))

    connection.commit()
    connection.close()


def get_transactions(user_id=None):
    """Return transaction history.

    If user_id is provided, only that user's transactions are returned.
    """

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    if user_id:
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
                message,
                user_id,
                recommended_action
            FROM transactions
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,))
    else:
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
                message,
                user_id,
                recommended_action
            FROM transactions
            ORDER BY id DESC
        """)

    rows = cursor.fetchall()

    connection.close()

    return rows


def save_feedback(transaction_id, feedback):
    """
    Save user feedback for a fraud alert.

    Allowed feedback values:
        confirmed_fraud
        false_positive
    """

    if feedback not in [
        "confirmed_fraud",
        "false_positive"
    ]:
        raise ValueError(
            "Feedback must be 'confirmed_fraud' "
            "or 'false_positive'"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO fraud_feedback (
            transaction_id,
            feedback,
            timestamp
        )
        VALUES (?, ?, ?)
    """, (
        transaction_id,
        feedback,
        datetime.now().isoformat(timespec="seconds")
    ))

    connection.commit()
    connection.close()


def get_feedback():
    """Return all recorded fraud feedback."""

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            transaction_id,
            feedback,
            timestamp
        FROM fraud_feedback
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows