import pandas as pd


def explain_transaction(transaction, anomaly_score, fraud_prediction):
    """
    Generate an interpretable explanation for an Isolation Forest result.

    This does not fabricate SHAP percentages.
    It uses observable transaction values and the actual model result.
    """

    transaction_df = pd.DataFrame(transaction)

    explanations = []

    # Amount-based explanation
    if "Amount" in transaction_df.columns:
        amount = float(transaction_df.iloc[0]["Amount"])

        if amount > 1000:
            explanations.append(
                f"High transaction amount detected: {amount:.2f}"
            )
        else:
            explanations.append(
                f"Transaction amount is {amount:.2f}."
            )

    # Actual Isolation Forest anomaly score
    if anomaly_score < 0:
        explanations.append(
            "The transaction has an unusual pattern compared with "
            "the patterns learned from normal transactions."
        )
    else:
        explanations.append(
            "The transaction pattern is similar to normal transactions."
        )

    # Actual model classification
    if fraud_prediction == 1:
        explanations.append(
            "Isolation Forest classified this transaction as an anomaly."
        )
    else:
        explanations.append(
            "Isolation Forest did not classify this transaction as an anomaly."
        )

    return explanations