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

# Initialize SQLite database
init_database()


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Raghavi Fraud Detection API is running"
    }), 200


# ============================================================
# REAL-TIME FRAUD DETECTION
# ============================================================

@app.route("/api/fraud/check", methods=["POST"])
def check_fraud():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "error": "No transaction data provided"
            }), 400


        # ------------------------------------------------------
        # Accept either a transaction object or a single-item list
        # ------------------------------------------------------

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
                    "error": "Exactly one transaction must be provided"
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
                "error": "Request must contain a transaction object"
            }), 400


        # ------------------------------------------------------
        # Convert transaction to DataFrame
        # ------------------------------------------------------

        transaction = pd.DataFrame(
            [transaction_data]
        )


        # ------------------------------------------------------
        # Required model features
        # ------------------------------------------------------

        expected_features = [
            "Time",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13",
            "V14", "V15", "V16", "V17", "V18", "V19",
            "V20", "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]


        # ------------------------------------------------------
        # Check missing features
        # ------------------------------------------------------

        missing_features = [
            feature
            for feature in expected_features
            if feature not in transaction.columns
        ]

        if missing_features:

            return jsonify({
                "status": "error",
                "error": "Missing required transaction features",
                "missing_features": missing_features
            }), 400


        # ------------------------------------------------------
        # Check unexpected features
        # ------------------------------------------------------

        extra_features = [
            feature
            for feature in transaction.columns
            if feature not in expected_features
        ]

        if extra_features:

            return jsonify({
                "status": "error",
                "error": "Unexpected transaction features",
                "extra_features": extra_features
            }), 400


        # ------------------------------------------------------
        # Ensure exact model feature order
        # ------------------------------------------------------

        transaction = transaction[
            expected_features
        ]


        # ------------------------------------------------------
        # Real-time fraud detection
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # Transaction information
        # ------------------------------------------------------

        amount = float(
            transaction.iloc[0]["Amount"]
        )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )


        # ------------------------------------------------------
        # Explainability
        # ------------------------------------------------------

        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )


        # ------------------------------------------------------
        # Instant alert
        # ------------------------------------------------------

        if (
            fraud_prediction == 1
            or fraud_risk_level == "High"
        ):

            alert = True

            reason = (
                "The transaction has been classified as "
                "potentially fraudulent or has a high fraud risk."
            )

            recommended_action = (
                "Review the transaction immediately and verify "
                "the transaction with the account holder before "
                "approving it."
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
                "Monitor the transaction and verify it if "
                "other suspicious activity is observed."
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


        # ------------------------------------------------------
        # Save transaction
        # ------------------------------------------------------

        save_transaction(
            user_id=user_id,
            amount=amount,
            fraud_prediction=fraud_prediction,
            anomaly_score=anomaly_score,
            fraud_risk_score=fraud_risk_score,
            fraud_risk_level=fraud_risk_level,
            alert=alert,
            message=message,
            recommended_action=recommended_action
        )


        # ------------------------------------------------------
        # Get ID of the transaction just saved
        # ------------------------------------------------------

        transactions = get_transactions(
            user_id=user_id
        )

        transaction_id = None

        if transactions:

            transaction_id = transactions[0][0]


        # ------------------------------------------------------
        # Return complete response
        # ------------------------------------------------------

        return jsonify({

            "status": "success",

            "user_id": user_id,

            "transaction_id": transaction_id,

            "transaction": {
                "amount": amount,
                "time": timestamp
            },

            "fraud_prediction": fraud_prediction,

            "anomaly_score": anomaly_score,

            "fraud_risk_score": fraud_risk_score,

            "fraud_risk_level": fraud_risk_level,

            "alert": alert,

            "alert_details": {

                "amount": amount,

                "time": timestamp,

                "risk_score": fraud_risk_score,

                "risk_level": fraud_risk_level,

                "reason": reason,

                "anomaly_score": anomaly_score,

                "recommended_action":
                    recommended_action
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


# ============================================================
# TRANSACTION HISTORY
# ============================================================

@app.route(
    "/api/fraud/history",
    methods=["GET"]
)
def fraud_history():

    try:

        # Optional:
        # /api/fraud/history?user_id=raghavi_demo

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

                "user_id": transaction[1],

                "timestamp": transaction[2],

                "amount": transaction[3],

                "fraud_prediction": transaction[4],

                "anomaly_score": transaction[5],

                "fraud_risk_score": transaction[6],

                "fraud_risk_level": transaction[7],

                "alert": bool(
                    transaction[8]
                ),

                "message": transaction[9],

                "recommended_action":
                    transaction[10]
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


# ============================================================
# FEEDBACK LOOP
# ============================================================

@app.route(
    "/api/fraud/feedback",
    methods=["POST"]
)
def submit_feedback():

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


        # ------------------------------------------------------
        # Validate transaction ID
        # ------------------------------------------------------

        if transaction_id is None:

            return jsonify({

                "status": "error",

                "error":
                    "transaction_id is required"

            }), 400


        try:

            transaction_id = int(
                transaction_id
            )

        except (TypeError, ValueError):

            return jsonify({

                "status": "error",

                "error":
                    "transaction_id must be an integer"

            }), 400


        # ------------------------------------------------------
        # Validate feedback
        # ------------------------------------------------------

        allowed_feedback = [
            "confirmed_fraud",
            "false_positive"
        ]

        if feedback not in allowed_feedback:

            return jsonify({

                "status": "error",

                "error":
                    "feedback must be "
                    "'confirmed_fraud' or "
                    "'false_positive'"

            }), 400


        # ------------------------------------------------------
        # Save feedback
        # ------------------------------------------------------

        save_feedback(
            transaction_id,
            feedback
        )


        return jsonify({

            "status": "success",

            "message":
                "Fraud feedback saved successfully.",

            "transaction_id":
                transaction_id,

            "feedback":
                feedback

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


# ============================================================
# FEEDBACK HISTORY
# ============================================================

@app.route(
    "/api/fraud/feedback",
    methods=["GET"]
)
def feedback_history():

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


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("========================================")

    print(
        "RAGHAVI FRAUD DETECTION API"
    )

    print("========================================")

    print(
        "\nStarting Flask server..."
    )

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )