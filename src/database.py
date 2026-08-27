import sqlite3
import os
from datetime import datetime


# =========================================================
# DATABASE LOCATION
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "fraud_transactions.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    """Create a connection to the SQLite database."""

    connection = sqlite3.connect(DATABASE_PATH)

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def init_database():
    """Create all required database tables."""

    connection = get_connection()
    cursor = connection.cursor()

    # -----------------------------------------------------
    # Main transaction table
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL,
            fraud_prediction INTEGER NOT NULL,
            anomaly_score REAL NOT NULL,
            fraud_risk_score REAL NOT NULL,
            fraud_risk_level TEXT NOT NULL,
            alert INTEGER NOT NULL,
            message TEXT NOT NULL,
            user_id TEXT,
            recommended_action TEXT
        )
    """)

    # -----------------------------------------------------
    # Fraud feedback table
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # -----------------------------------------------------
    # Index for faster timestamp searches
    # -----------------------------------------------------

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
        ON transactions(timestamp)
    """)

    # -----------------------------------------------------
    # Index for faster user searches
    # -----------------------------------------------------

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_user_id
        ON transactions(user_id)
    """)

    connection.commit()
    connection.close()


# =========================================================
# SAVE TRANSACTION
# =========================================================

def save_transaction(
    amount,
    fraud_prediction,
    anomaly_score,
    fraud_risk_score,
    fraud_risk_level,
    alert,
    message,
    user_id=None,
    recommended_action=None,
    timestamp=None
):
    """Save a fraud detection result."""

    if timestamp is None:

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

    connection = get_connection()
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
        timestamp,
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

    transaction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return transaction_id


# =========================================================
# GET ALL TRANSACTIONS
# =========================================================

def get_transactions(user_id=None):
    """
    Return transaction history.

    If user_id is provided,
    only that user's transactions are returned.
    """

    connection = get_connection()
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


# =========================================================
# GET TRANSACTIONS BY TIME RANGE
# =========================================================

def get_transactions_by_time_range(
    start_time,
    end_time,
    user_id=None
):
    """
    Return transactions between start_time and end_time.

    Example:

    start_time = 2026-08-27T19:05:00
    end_time   = 2026-08-27T19:06:00
    """

    connection = get_connection()
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
            WHERE timestamp >= ?
              AND timestamp < ?
              AND user_id = ?
            ORDER BY timestamp ASC
        """, (
            start_time,
            end_time,
            user_id
        ))

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
            WHERE timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp ASC
        """, (
            start_time,
            end_time
        ))

    rows = cursor.fetchall()

    connection.close()

    return rows


# =========================================================
# SAVE FEEDBACK
# =========================================================

def save_feedback(transaction_id, feedback):
    """
    Save user feedback.

    Allowed feedback:

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

    connection = get_connection()
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
        datetime.now().isoformat(
            timespec="seconds"
        )
    ))

    connection.commit()
    connection.close()


# =========================================================
# GET FEEDBACK
# =========================================================

def get_feedback():
    """Return all recorded fraud feedback."""

    connection = get_connection()
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