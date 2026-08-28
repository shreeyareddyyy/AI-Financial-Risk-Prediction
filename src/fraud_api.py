from flask import Flask, request, jsonify
from datetime import datetime
import sqlite3
import os

from fraud_detection import predict_transaction
from explainability import explain_transaction


app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    BASE_DIR,
    "fraud_transactions.db"
)


# =========================================================
# SAVE TRANSACTION
# =========================================================

def save_transaction(transaction, result, alert):

    amount = float(
        transaction.get("Amount", 0)
    )

    fraud_prediction = int(
        result.iloc[0]["Fraud_Prediction"]
    )

    risk_score = float(
        result.iloc[0]["Fraud_Risk_Score"]
    )

    risk_level = str(
        result.iloc[0]["Fraud_Risk_Level"]
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.execute(
        """
        INSERT INTO fraud_transactions
        (
            timestamp,
            amount,
            fraud_prediction,
            risk_score,
            risk_level,
            alert
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            amount,
            fraud_prediction,
            risk_score,
            risk_level,
            int(alert)
        )
    )

    transaction_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return transaction_id


# =========================================================
# FRAUD CHECK API
# =========================================================

@app.route(
    "/api/fraud/check",
    methods=["POST"]
)
def check_fraud():

    try:

        transaction = request.get_json()

        if not transaction:

            return jsonify({
                "error":
                    "No transaction data provided."
            }), 400

        # -------------------------------------------------
        # FRAUD PREDICTION
        # -------------------------------------------------

        result = predict_transaction(
            transaction
        )

        fraud_prediction = int(
            result.iloc[0]["Fraud_Prediction"]
        )

        anomaly_score = float(
            result.iloc[0]["Anomaly_Score"]
        )

        risk_score = float(
            result.iloc[0]["Fraud_Risk_Score"]
        )

        risk_level = str(
            result.iloc[0]["Fraud_Risk_Level"]
        )

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )

        # -------------------------------------------------
        # ALERT
        # -------------------------------------------------

        alert = risk_level == "High"

        if alert:

            message = (
                "High fraud risk detected. "
                "Review this transaction."
            )

            recommended_action = (
                "Block transaction and "
                "review immediately."
            )

        elif risk_level == "Medium":

            message = (
                "Medium fraud risk detected. "
                "Review if necessary."
            )

            recommended_action = (
                "Review the transaction "
                "before taking action."
            )

        else:

            message = (
                "Transaction appears to have "
                "low fraud risk."
            )

            recommended_action = (
                "Transaction can proceed; "
                "continue monitoring."
            )

        # -------------------------------------------------
        # SAVE TRANSACTION
        # -------------------------------------------------

        transaction_id = save_transaction(
            transaction,
            result,
            alert
        )

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "transaction_id":
                transaction_id,

            "amount":
                float(
                    transaction.get(
                        "Amount",
                        0
                    )
                ),

            "time":
                transaction.get("Time"),

            "fraud_prediction":
                fraud_prediction,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "alert":
                alert,

            "message":
                message,

            "reason":
                explanation,

            "anomaly_score":
                anomaly_score,

            "recommended_action":
                recommended_action
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# USER FEEDBACK
# =========================================================

@app.route(
    "/api/fraud/feedback",
    methods=["POST"]
)
def fraud_feedback():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error":
                    "No feedback data provided."
            }), 400

        transaction_id = data.get(
            "transaction_id"
        )

        feedback = data.get(
            "feedback"
        )

        if transaction_id is None:

            return jsonify({
                "error":
                    "transaction_id is required."
            }), 400

        if feedback not in [
            "confirmed fraud",
            "false positive"
        ]:

            return jsonify({
                "error":
                    "Feedback must be "
                    "'confirmed fraud' or "
                    "'false positive'."
            }), 400

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.execute(
            """
            UPDATE fraud_transactions
            SET feedback = ?
            WHERE id = ?
            """,
            (
                feedback,
                transaction_id
            )
        )

        conn.commit()

        if cursor.rowcount == 0:

            conn.close()

            return jsonify({
                "error":
                    "Transaction not found."
            }), 404

        conn.close()

        return jsonify({

            "success":
                True,

            "transaction_id":
                transaction_id,

            "feedback":
                feedback,

            "message":
                "Feedback recorded successfully."
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# TRANSACTION HISTORY
# =========================================================

@app.route(
    "/api/fraud/history",
    methods=["GET"]
)
def fraud_history():

    try:

        conn = sqlite3.connect(
            DB_PATH
        )

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                timestamp,
                amount,
                fraud_prediction,
                risk_score,
                risk_level,
                alert,
                feedback
            FROM fraud_transactions
            ORDER BY id DESC
            """
        ).fetchall()

        conn.close()

        history = [
            dict(row)
            for row in rows
        ]

        return jsonify({
            "transactions":
                history
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )