from datetime import datetime

from flask import Flask, request, jsonify
import pandas as pd

from fraud_detection import predict_transaction
from database import init_database, save_transaction, get_transactions
from explainability import explain_transaction


app = Flask(__name__)

# Initialize SQLite database
init_database()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Raghavi Fraud Detection API is running"
    })


@app.route("/api/fraud/check", methods=["POST"])
def check_fraud():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "error": "No transaction data provided"
            }), 400

        # ---------------------------------------------------------
        # Accept either:
        # {
        #     "user_id": "user123",
        #     "Time": ...,
        #     "V1": ...,
        #     ...
        # }
        #
        # or a list containing one transaction.
        # ---------------------------------------------------------

        if isinstance(data, dict):
            user_id = data.get("user_id", "demo_user")

            transaction_data = {
                key: value
                for key, value in data.items()
                if key != "user_id"
            }

        elif isinstance(data, list):

            if len(data) != 1 or not isinstance(data[0], dict):
                return jsonify({
                    "status": "error",
                    "error": "Exactly one transaction must be provided"
                }), 400

            user_id = data[0].get("user_id", "demo_user")

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

        # ---------------------------------------------------------
        # Convert transaction into DataFrame
        # ---------------------------------------------------------

        transaction = pd.DataFrame([transaction_data])

        # ---------------------------------------------------------
        # Required model features
        # ---------------------------------------------------------

        expected_features = [
            "Time",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13",
            "V14", "V15", "V16", "V17", "V18", "V19",
            "V20", "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]

        missing_features = [
            feature
            for feature in expected_features
            if feature not in transaction.columns
        ]

        extra_features = [
            feature
            for feature in transaction.columns
            if feature not in expected_features
        ]

        if missing_features:
            return jsonify({
                "status": "error",
                "error": "Missing required transaction features",
                "missing_features": missing_features
            }), 400

        if extra_features:
            return jsonify({
                "status": "error",
                "error": "Unexpected transaction features",
                "extra_features": extra_features
            }), 400

        # Ensure exact model feature order
        transaction = transaction[expected_features]

        # ---------------------------------------------------------
        # Real-time fraud detection
        # ---------------------------------------------------------

        result = predict_transaction(transaction)

        row = result.iloc[0]

        fraud_prediction = int(row["Fraud_Prediction"])
        anomaly_score = float(row["Anomaly_Score"])
        fraud_risk_score = float(row["Fraud_Risk_Score"])
        fraud_risk_level = str(row["Fraud_Risk_Level"])

        # ---------------------------------------------------------
        # Transaction details
        # ---------------------------------------------------------

        amount = float(transaction.iloc[0]["Amount"])

        # Human-readable timestamp for the API alert
        timestamp = datetime.now().isoformat(timespec="seconds")

        # ---------------------------------------------------------
        # Explainability
        # ---------------------------------------------------------

        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )

        # ---------------------------------------------------------
        # Instant alert
        # ---------------------------------------------------------

        if fraud_prediction == 1 or fraud_risk_level == "High":

            alert = True

            reason = (
                "The transaction has been classified as potentially "
                "fraudulent or has a high fraud risk."
            )

            recommended_action = (
                "Review the transaction immediately and verify the "
                "transaction with the account holder before approving it."
            )

            message = (
                "Transaction flagged as potentially fraudulent."
            )

        elif fraud_risk_level == "Medium":

            alert = False

            reason = (
                "The transaction has a medium fraud risk and should "
                "be monitored."
            )

            recommended_action = (
                "Monitor the transaction and verify it if other "
                "suspicious activity is observed."
            )

            message = (
                "Transaction has medium fraud risk."
            )

        else:

            alert = False

            reason = (
                "The transaction has a low fraud risk and the model "
                "did not classify it as an anomaly."
            )

            recommended_action = (
                "No immediate action required."
            )

            message = (
                "Transaction appears normal."
            )

        # ---------------------------------------------------------
        # Save transaction in SQLite
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Return complete real-time response
        # ---------------------------------------------------------

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

            "alert_details": {
                "amount": amount,
                "time": timestamp,
                "risk_score": fraud_risk_score,
                "risk_level": fraud_risk_level,
                "reason": reason,
                "anomaly_score": anomaly_score,
                "recommended_action": recommended_action
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


@app.route("/api/fraud/history", methods=["GET"])
def fraud_history():

    try:

        # Optional user filtering:
        # /api/fraud/history?user_id=user123

        user_id = request.args.get("user_id")

        transactions = get_transactions(user_id=user_id)

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
                "alert": bool(transaction[8]),
                "message": transaction[9],
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