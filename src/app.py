from flask import Flask, request, jsonify
import pandas as pd

from fraud_detection import predict_transaction
from database import init_database, save_transaction, get_transactions
from explainability import explain_transaction


app = Flask(__name__)

# Create SQLite database
init_database()


@app.route("/")
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
                "error": "No transaction data provided"
            }), 400

        transaction = pd.DataFrame(data)

        # Run fraud detection
        result = predict_transaction(transaction)

        row = result.iloc[0]

        fraud_prediction = int(row["Fraud_Prediction"])
        anomaly_score = float(row["Anomaly_Score"])
        fraud_risk_score = float(row["Fraud_Risk_Score"])
        fraud_risk_level = str(row["Fraud_Risk_Level"])

        # Get transaction amount
        amount = None

        if "Amount" in transaction.columns:
            amount = float(transaction.iloc[0]["Amount"])

        # Generate explanation
        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )

        # Alert and message
        if fraud_prediction == 1 or fraud_risk_level == "High":

            alert = True
            message = "Transaction flagged as potentially fraudulent."

        else:

            alert = False
            message = "Transaction appears normal."

        # Save transaction
        save_transaction(
            amount=amount,
            fraud_prediction=fraud_prediction,
            anomaly_score=anomaly_score,
            fraud_risk_score=fraud_risk_score,
            fraud_risk_level=fraud_risk_level,
            alert=alert,
            message=message
        )

        # Return API response
        return jsonify({
            "fraud_prediction": fraud_prediction,
            "anomaly_score": anomaly_score,
            "fraud_risk_score": fraud_risk_score,
            "fraud_risk_level": fraud_risk_level,
            "alert": alert,
            "message": message,
            "explanation": explanation
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/fraud/history", methods=["GET"])
def fraud_history():

    try:
        transactions = get_transactions()

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
                "alert": bool(transaction[7]),
                "message": transaction[8]
            })

        return jsonify({
            "status": "success",
            "count": len(results),
            "transactions": results
        })

    except Exception as e:

        return jsonify({
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