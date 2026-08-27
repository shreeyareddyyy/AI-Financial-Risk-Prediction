from datetime import datetime, timedelta

from flask import Flask, request, jsonify
import pandas as pd

from fraud_detection import predict_transaction

from database import (
    init_database,
    save_transaction,
    get_transactions,
    get_transactions_by_time_range,
    save_feedback,
    get_feedback
)

from explainability import explain_transaction


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_database()


# =========================================================
# RAPID TRANSACTION DETECTION
# =========================================================

def check_rapid_transactions(
    user_id,
    window_seconds=60,
    threshold=3
):
    """
    Detect rapid transaction activity.

    If 3 or more transactions from the same user
    occur within 60 seconds, rapid activity is detected.
    """

    if not user_id:

        return False, 1

    transactions = get_transactions(
        user_id=user_id
    )

    now = datetime.now()

    previous_count = 0

    for transaction in transactions:

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

    total_recent_count = previous_count + 1

    rapid_activity = (
        total_recent_count >= threshold
    )

    return rapid_activity, total_recent_count


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Raghavi Fraud Detection API is running",
        "features": [
            "Real-time fraud detection",
            "Explainable fraud alerts",
            "Rapid transaction detection",
            "Transaction history",
            "Fraud feedback",
            "Daily fraud analysis",
            "Hourly fraud analysis",
            "Minute-level fraud analysis",
            "Second-level activity monitoring"
        ]
    })


# =========================================================
# FRAUD DETECTION
# =========================================================

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
        # READ USER ID
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
        # CREATE DATAFRAME
        # -------------------------------------------------

        transaction = pd.DataFrame(
            [transaction_data]
        )

        # -------------------------------------------------
        # REQUIRED MODEL FEATURES
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
        # CHECK MISSING FEATURES
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
        # CHECK UNEXPECTED FEATURES
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
        # EXACT MODEL ORDER
        # -------------------------------------------------

        transaction = transaction[
            expected_features
        ]

        # -------------------------------------------------
        # ML FRAUD DETECTION
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
        # TRANSACTION INFORMATION
        # -------------------------------------------------

        amount = float(
            transaction.iloc[0]["Amount"]
        )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        current_datetime = datetime.fromisoformat(
            timestamp
        )

        # -------------------------------------------------
        # TIME COMPONENTS
        # -------------------------------------------------

        transaction_date = (
            current_datetime.strftime("%Y-%m-%d")
        )

        transaction_hour = (
            current_datetime.hour
        )

        transaction_minute = (
            current_datetime.minute
        )

        transaction_second = (
            current_datetime.second
        )

        # -------------------------------------------------
        # EXPLAINABILITY
        # -------------------------------------------------

        explanation = explain_transaction(
            transaction,
            anomaly_score,
            fraud_prediction
        )

        # -------------------------------------------------
        # RAPID TRANSACTION DETECTION
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
        # ALERT DECISION
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
        # SAVE TRANSACTION
        # -------------------------------------------------

        transaction_id = save_transaction(
            amount=amount,
            fraud_prediction=fraud_prediction,
            anomaly_score=anomaly_score,
            fraud_risk_score=fraud_risk_score,
            fraud_risk_level=fraud_risk_level,
            alert=alert,
            message=message,
            user_id=user_id,
            recommended_action=recommended_action,
            timestamp=timestamp
        )

        # -------------------------------------------------
        # API RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "transaction_id": transaction_id,

            "user_id": user_id,

            "transaction": {

                "amount": amount,

                "time": timestamp,

                "date": transaction_date,

                "hour": transaction_hour,

                "minute": transaction_minute,

                "second": transaction_second
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


# =========================================================
# TRANSACTION HISTORY
# =========================================================

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


# =========================================================
# TIME ANALYSIS HELPER
# =========================================================

def create_time_summary(
    rows,
    period
):
    """
    Create fraud statistics for a selected time period.
    """

    total_transactions = len(rows)

    fraud_alerts = sum(
        1 for row in rows
        if int(row[7]) == 1
    )

    ml_fraud_predictions = sum(
        1 for row in rows
        if int(row[3]) == 1
    )

    normal_transactions = (
        total_transactions - fraud_alerts
    )

    if total_transactions > 0:

        fraud_rate = (
            fraud_alerts
            / total_transactions
        ) * 100

    else:

        fraud_rate = 0.0

    return {

        "period": period,

        "total_transactions": (
            total_transactions
        ),

        "fraud_alerts": fraud_alerts,

        "ml_fraud_predictions": (
            ml_fraud_predictions
        ),

        "normal_transactions": (
            normal_transactions
        ),

        "fraud_rate_percent": round(
            fraud_rate,
            2
        )
    }


# =========================================================
# TIME ANALYSIS
# =========================================================

@app.route(
    "/api/fraud/time-analysis",
    methods=["GET"]
)
def fraud_time_analysis():

    try:

        date_string = request.args.get(
            "date"
        )

        hour_string = request.args.get(
            "hour"
        )

        minute_string = request.args.get(
            "minute"
        )

        second_string = request.args.get(
            "second"
        )

        user_id = request.args.get(
            "user_id"
        )

        # -------------------------------------------------
        # DATE REQUIRED
        # -------------------------------------------------

        if not date_string:

            return jsonify({

                "status": "error",

                "error": (
                    "date is required. "
                    "Use YYYY-MM-DD."
                )

            }), 400

        # -------------------------------------------------
        # VALIDATE DATE
        # -------------------------------------------------

        try:

            selected_date = datetime.strptime(
                date_string,
                "%Y-%m-%d"
            )

        except ValueError:

            return jsonify({

                "status": "error",

                "error": (
                    "Invalid date format. "
                    "Use YYYY-MM-DD."
                )

            }), 400

        # -------------------------------------------------
        # DAILY ANALYSIS
        # -------------------------------------------------

        day_start = selected_date

        day_end = (
            day_start
            + timedelta(days=1)
        )

        daily_rows = get_transactions_by_time_range(
            day_start.isoformat(
                timespec="seconds"
            ),
            day_end.isoformat(
                timespec="seconds"
            ),
            user_id
        )

        daily_summary = create_time_summary(
            daily_rows,
            "daily"
        )

        # -------------------------------------------------
        # HOURLY ANALYSIS
        # -------------------------------------------------

        hourly_analysis = []

        for hour in range(24):

            hour_start = (
                selected_date
                + timedelta(hours=hour)
            )

            hour_end = (
                hour_start
                + timedelta(hours=1)
            )

            rows = get_transactions_by_time_range(
                hour_start.isoformat(
                    timespec="seconds"
                ),
                hour_end.isoformat(
                    timespec="seconds"
                ),
                user_id
            )

            summary = create_time_summary(
                rows,
                f"{hour:02d}:00"
            )

            hourly_analysis.append(
                summary
            )

        # -------------------------------------------------
        # OPTIONAL SPECIFIC HOUR
        # -------------------------------------------------

        selected_hour_analysis = None

        if hour_string is not None:

            try:

                hour = int(hour_string)

                if hour < 0 or hour > 23:

                    raise ValueError

            except ValueError:

                return jsonify({

                    "status": "error",

                    "error": (
                        "hour must be between 0 and 23."
                    )

                }), 400

            hour_start = (
                selected_date
                + timedelta(hours=hour)
            )

            hour_end = (
                hour_start
                + timedelta(hours=1)
            )

            rows = get_transactions_by_time_range(
                hour_start.isoformat(
                    timespec="seconds"
                ),
                hour_end.isoformat(
                    timespec="seconds"
                ),
                user_id
            )

            selected_hour_analysis = (
                create_time_summary(
                    rows,
                    f"{hour:02d}:00"
                )
            )

        # -------------------------------------------------
        # OPTIONAL SPECIFIC MINUTE
        # -------------------------------------------------

        selected_minute_analysis = None

        if (
            hour_string is not None
            and minute_string is not None
        ):

            try:

                hour = int(hour_string)
                minute = int(minute_string)

                if (
                    hour < 0
                    or hour > 23
                    or minute < 0
                    or minute > 59
                ):

                    raise ValueError

            except ValueError:

                return jsonify({

                    "status": "error",

                    "error": (
                        "hour must be 0-23 "
                        "and minute must be 0-59."
                    )

                }), 400

            minute_start = (
                selected_date
                + timedelta(
                    hours=hour,
                    minutes=minute
                )
            )

            minute_end = (
                minute_start
                + timedelta(minutes=1)
            )

            rows = get_transactions_by_time_range(
                minute_start.isoformat(
                    timespec="seconds"
                ),
                minute_end.isoformat(
                    timespec="seconds"
                ),
                user_id
            )

            selected_minute_analysis = (
                create_time_summary(
                    rows,
                    minute_start.strftime(
                        "%H:%M"
                    )
                )
            )

        # -------------------------------------------------
        # OPTIONAL SPECIFIC SECOND
        # -------------------------------------------------

        selected_second_analysis = None

        if (
            hour_string is not None
            and minute_string is not None
            and second_string is not None
        ):

            try:

                hour = int(hour_string)
                minute = int(minute_string)
                second = int(second_string)

                if (
                    hour < 0
                    or hour > 23
                    or minute < 0
                    or minute > 59
                    or second < 0
                    or second > 59
                ):

                    raise ValueError

            except ValueError:

                return jsonify({

                    "status": "error",

                    "error": (
                        "Invalid hour, minute, "
                        "or second."
                    )

                }), 400

            second_start = (
                selected_date
                + timedelta(
                    hours=hour,
                    minutes=minute,
                    seconds=second
                )
            )

            second_end = (
                second_start
                + timedelta(seconds=1)
            )

            rows = get_transactions_by_time_range(
                second_start.isoformat(
                    timespec="seconds"
                ),
                second_end.isoformat(
                    timespec="seconds"
                ),
                user_id
            )

            selected_second_analysis = (
                create_time_summary(
                    rows,
                    second_start.strftime(
                        "%H:%M:%S"
                    )
                )
            )

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "date": date_string,

            "user_id": user_id,

            "daily_analysis": daily_summary,

            "hourly_analysis": hourly_analysis,

            "selected_hour": (
                selected_hour_analysis
            ),

            "selected_minute": (
                selected_minute_analysis
            ),

            "selected_second": (
                selected_second_analysis
            )

        }), 200

    except Exception as e:

        return jsonify({

            "status": "error",

            "error": str(e)

        }), 500


# =========================================================
# FRAUD FEEDBACK - SAVE
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

                "status": "error",

                "error": (
                    "No feedback data provided"
                )

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

                "error": (
                    "transaction_id is required"
                )

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


# =========================================================
# FRAUD FEEDBACK - HISTORY
# =========================================================

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


# =========================================================
# START FLASK SERVER
# =========================================================

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