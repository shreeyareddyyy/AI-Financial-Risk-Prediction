from datetime import datetime

from flask import Flask, request, jsonify
import pandas as pd

from fraud_detection import predict_transaction
from database import (
    init_database,
    save_transaction,
    get_transactions,
    save_feedback,
    get_feedback
)
from explainability import explain_transaction


app = Flask(__name__)

# ---------------------------------------------------------
# Initialize database
# ---------------------------------------------------------

init_database()


# ---------------------------------------------------------
# Rapid transaction detection
# ---------------------------------------------------------

def check_rapid_transactions(
    user_id,
    window_seconds=60,
    threshold=3
):
    """
    Detect rapid transaction activity.

    If 3 or more transactions from the same user occur
    within 60 seconds, rapid activity is detected.

    The current transaction is included in the count.
    """

    if not user_id:
        return False, 1

    transactions = get_transactions(
        user_id=user_id
    )

    now = datetime.now()

    previous_count = 0

    for transaction in transactions:

        # database.py returns:
        # id, timestamp, amount, fraud_prediction,
        # anomaly_score, fraud_risk_score,
        # fraud_risk_level, alert, message,
        # user_id, recommended_action

        timestamp = transaction[1]

        try:
            transaction_time = datetime.fromisoformat(
                timestamp
            )
        except (ValueError, TypeError):
            continue

        time_difference = (
            now - transaction_time
        ).total_seconds()

        if 0 <= time_difference <= window_seconds:
            previous_count += 1

    # Include the current transaction
    total_recent_count = previous_count + 1

    rapid_activity = (
        total_recent_count >= threshold
    )

    return rapid_activity, total_recent_count


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Raghavi Fraud Detection API is running"
    })


# ---------------------------------------------------------
# Fraud Detection
# ---------------------------------------------------------

@app.route(
    "/api/fraud/check",
    methods=["POST"]
)
def check_fraud():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "error": "No transaction data provided"
            }), 400

        # -------------------------------------------------
        # Read user ID
        # -------------------------------------------------

        if isinstance(data, dict):

            user_id = data.get(
                "user_id",
                "demo_user"
            )

            transaction_data = {
                key: value
                for key, value in data.items()
                if key != "user_id"
            }

        elif isinstance(data, list):

            if (
                len(data) != 1
                or not isinstance(data[0], dict)
            ):

                return jsonify({
                    "status": "error",
                    "error": (
                        "Exactly one transaction "
                        "must be provided"
                    )
                }), 400

            user_id = data[0].get(
                "user_id",
                "demo_user"
            )

            transaction_data = {
                key: value
                for key, value in data[0].items()
                if key != "user_id"
            }

        else:

            return jsonify({
                "status": "error",
                "error": (
                    "Request must contain "
                    "a transaction object"
                )
            }), 400

        # -------------------------------------------------
        # Convert transaction to DataFrame
        # -------------------------------------------------

        transaction = pd.DataFrame(
            [transaction_data]
        )

        # -------------------------------------------------
        # Required model features
        # -------------------------------------------------

        expected_features = [
            "Time",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13",
            "V14", "V15", "V16", "V17", "V18", "V19",
            "V20", "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]

        # -------------------------------------------------
        # Check missing features
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in expected_features
            if feature not in transaction.columns
        ]

        if missing_features:

            return jsonify({
                "status": "error",
                "error": (
                    "Missing required transaction features"
                ),
                "missing_features": missing_features
            }), 400

        # -------------------------------------------------
        # Check unexpected features
        # -------------------------------------------------

        extra_features = [
            feature
            for feature in transaction.columns
            if feature not in expected_features
        ]

        if extra_features:

            return jsonify({
                "status": "error",
                "error": (
                    "Unexpected transaction features"
                ),
                "extra_features": extra_features
            }), 400

        # -------------------------------------------------
        # Put features in exact model order
        # -------------------------------------------------

        transaction = transaction[
            expected_features
        ]

        # -------------------------------------------------
        # ML fraud detection
        # -------------------------------------------------

        result = predict_transaction(
            transaction
        )

        row = result.iloc[0]

        fraud_prediction = int(
            row["Fraud_Prediction"]
        )

        anomaly_score = float(
            row["Anomaly_Score"]
        )

        fraud_risk_score = float(
            row["Fraud_Risk_Score"]
        )

        fraud_risk_level = str(
            row["Fraud_Risk_Level"]
        )

        # -------------------------------------------------
        # Transaction information
        # -------------------------------------------------

        amount = float(
            transaction.iloc[0]["Amount"]
        )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        # -------------------------------------------------
        # Explainability
        # -------------------------------------------------

        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )

        # -------------------------------------------------
        # Rapid transaction detection
        # -------------------------------------------------

        rapid_activity, recent_transaction_count = (
            check_rapid_transactions(
                user_id=user_id,
                window_seconds=60,
                threshold=3
            )
        )

        if rapid_activity:

            explanation.append(
                "Rapid transaction activity detected: "
                f"{recent_transaction_count} transactions "
                "within 60 seconds."
            )

        # -------------------------------------------------
        # Alert decision
        # -------------------------------------------------

        if rapid_activity:

            alert = True

            reason = (
                "Rapid transaction activity detected. "
                f"{recent_transaction_count} transactions "
                "occurred within 60 seconds."
            )

            recommended_action = (
                "Review the recent transactions immediately "
                "and verify the activity with the account holder."
            )

            message = (
                "Transaction flagged due to rapid "
                "transaction activity."
            )

        elif (
            fraud_prediction == 1
            or fraud_risk_level == "High"
        ):

            alert = True

            reason = (
                "The transaction has been classified as "
                "potentially fraudulent or has a high "
                "fraud risk."
            )

            recommended_action = (
                "Review the transaction immediately and "
                "verify the transaction with the account "
                "holder before approving it."
            )

            message = (
                "Transaction flagged as potentially fraudulent."
            )

        elif fraud_risk_level == "Medium":

            alert = False

            reason = (
                "The transaction has a medium fraud risk "
                "and should be monitored."
            )

            recommended_action = (
                "Monitor the transaction and verify it "
                "if other suspicious activity is observed."
            )

            message = (
                "Transaction has medium fraud risk."
            )

        else:

            alert = False

            reason = (
                "The transaction has a low fraud risk and "
                "the model did not classify it as an anomaly."
            )

            recommended_action = (
                "No immediate action required."
            )

            message = (
                "Transaction appears normal."
            )

        # -------------------------------------------------
        # Save transaction
        # -------------------------------------------------

        save_transaction(
            amount=amount,
            fraud_prediction=fraud_prediction,
            anomaly_score=anomaly_score,
            fraud_risk_score=fraud_risk_score,
            fraud_risk_level=fraud_risk_level,
            alert=alert,
            message=message,
            user_id=user_id,
            recommended_action=recommended_action
        )

        # -------------------------------------------------
        # API response
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "user_id": user_id,

            "transaction": {
                "amount": amount,
                "time": timestamp
            },

            "fraud_prediction": fraud_prediction,

            "anomaly_score": anomaly_score,

            "fraud_risk_score": fraud_risk_score,

            "fraud_risk_level": fraud_risk_level,

            "alert": alert,

            "rapid_transaction_alert": rapid_activity,

            "recent_transaction_count": (
                recent_transaction_count
            ),

            "alert_details": {

                "amount": amount,

                "time": timestamp,

                "risk_score": fraud_risk_score,

                "risk_level": fraud_risk_level,

                "reason": reason,

                "anomaly_score": anomaly_score,

                "recommended_action": (
                    recommended_action
                ),

                "rapid_transaction_alert": (
                    rapid_activity
                ),

                "recent_transaction_count": (
                    recent_transaction_count
                )
            },

            "message": message,

            "explanation": explanation

        }), 200

    except ValueError as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 400

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Transaction History
# ---------------------------------------------------------

@app.route(
    "/api/fraud/history",
    methods=["GET"]
)
def fraud_history():

    try:

        user_id = request.args.get(
            "user_id"
        )

        transactions = get_transactions(
            user_id=user_id
        )

        results = []

        for transaction in transactions:

            results.append({

                "id": transaction[0],

                "timestamp": transaction[1],

                "amount": transaction[2],

                "fraud_prediction": transaction[3],

                "anomaly_score": transaction[4],

                "fraud_risk_score": transaction[5],

                "fraud_risk_level": transaction[6],

                "alert": bool(
                    transaction[7]
                ),

                "message": transaction[8],

                "user_id": transaction[9],

                "recommended_action": transaction[10]

            })

        return jsonify({

            "status": "success",

            "user_id": user_id,

            "count": len(results),

            "transactions": results

        }), 200

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Save Fraud Feedback
# ---------------------------------------------------------

@app.route(
    "/api/fraud/feedback",
    methods=["POST"]
)
def fraud_feedback():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "error": "No feedback data provided"
            }), 400

        transaction_id = data.get(
            "transaction_id"
        )

        feedback = data.get(
            "feedback"
        )

        if transaction_id is None:

            return jsonify({
                "status": "error",
                "error": "transaction_id is required"
            }), 400

        if feedback not in [
            "confirmed_fraud",
            "false_positive"
        ]:

            return jsonify({
                "status": "error",
                "error": (
                    "Feedback must be "
                    "'confirmed_fraud' or "
                    "'false_positive'"
                )
            }), 400

        save_feedback(
            transaction_id=int(
                transaction_id
            ),
            feedback=feedback
        )

        return jsonify({

            "status": "success",

            "message": (
                "Fraud feedback saved successfully."
            ),

            "transaction_id": int(
                transaction_id
            ),

            "feedback": feedback

        }), 200

    except ValueError as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 400

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Get Fraud Feedback
# ---------------------------------------------------------

@app.route(
    "/api/fraud/feedback",
    methods=["GET"]
)
def fraud_feedback_history():

    try:

        feedback_rows = get_feedback()

        results = []

        for row in feedback_rows:

            results.append({

                "id": row[0],

                "transaction_id": row[1],

                "feedback": row[2],

                "timestamp": row[3]

            })

        return jsonify({

            "status": "success",

            "count": len(results),

            "feedback": results

        }), 200

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Start Flask server
# ---------------------------------------------------------

if __name__ == "__main__":

    print("========================================")
    print("RAGHAVI FRAUD DETECTION API")
    print("========================================")

    print("\nStarting Flask server...")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )