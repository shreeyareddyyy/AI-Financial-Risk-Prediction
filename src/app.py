from flask import Flask, request, jsonify
import pandas as pd
from fraud_detection import predict_transaction

app = Flask(__name__)


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

        result = predict_transaction(transaction)

        row = result.iloc[0]

        fraud_prediction = int(row["Fraud_Prediction"])
        anomaly_score = float(row["Anomaly_Score"])
        fraud_risk_score = float(row["Fraud_Risk_Score"])
        fraud_risk_level = str(row["Fraud_Risk_Level"])

        if fraud_prediction == 1:
            alert = True
            message = "Transaction flagged as potentially fraudulent."
        else:
            alert = False
            message = "Transaction appears normal."

        return jsonify({
            "fraud_prediction": fraud_prediction,
            "anomaly_score": anomaly_score,
            "fraud_risk_score": fraud_risk_score,
            "fraud_risk_level": fraud_risk_level,
            "alert": alert,
            "message": message
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)