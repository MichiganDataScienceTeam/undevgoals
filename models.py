def status_quo_model(X):
    """Predict the most recent value of each series.

    If no most recent value exists, predict 0."""

    def mostrecent(x):
        x = x[x.notnull()]
        if len(x) > 0:
            return x.iloc[-1]
        else:
            return 0
    preds = X.apply(mostrecent, axis=1)

    return preds
