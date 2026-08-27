import pandas as pd


def explain_transaction(transaction, anomaly_score, fraud_prediction):
    """
    Creates a simple explanation using the actual transaction
    values and the model's anomaly result.
    """

    transaction_df = pd.DataFrame(transaction)

    explanations = []

    # Check transaction amount
    if "Amount" in transaction_df.columns:
        amount = float(transaction_df.iloc[0]["Amount"])

        if amount > 1000:
            explanations.append(
                f"High transaction amount detected: {amount:.2f}"
            )

    # Explain anomaly score
    if anomaly_score < 0:
        explanations.append(
            "The transaction has an unusual pattern compared with normal transactions."
        )
    else:
        explanations.append(
            "The transaction pattern is similar to normal transactions."
        )

    # Final model explanation
    if fraud_prediction == 1:
        explanations.append(
            "Isolation Forest classified this transaction as an anomaly."
        )
    else:
        explanations.append(
            "Isolation Forest did not classify this transaction as an anomaly."
        )

    return explanations