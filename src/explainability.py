def explain_transaction(
    transaction,
    anomaly_score,
    fraud_prediction
):

    explanations = []


    # =====================================================
    # EXPLAIN ANOMALY SCORE
    # =====================================================

    score = float(anomaly_score)

    if score < 0:

        explanations.append(
            "The transaction has an unusual pattern "
            "compared with the patterns learned by "
            "the Isolation Forest model."
        )

    else:

        explanations.append(
            "The transaction pattern is similar to "
            "normal transactions."
        )


    # =====================================================
    # EXPLAIN MODEL PREDICTION
    # =====================================================

    prediction = int(fraud_prediction)

    if prediction == 1:

        explanations.append(
            "Isolation Forest classified this "
            "transaction as an anomaly."
        )

    else:

        explanations.append(
            "Isolation Forest did not classify this "
            "transaction as an anomaly."
        )


    return explanations