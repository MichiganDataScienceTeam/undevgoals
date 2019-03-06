import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.stattools import MissingDataError
import warnings

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

def arima(X, order = (2,2,0), backup_order = (1,1,0), lookback = 100, forward=5):

    """Predict the most recent value using an ARIMA model.

    By default, will fit ARIMA(1,1,1) for each row by using

    the 5 most recent years of the time series.

    If any issues fitting time series model, use status_quo."""
    
    sq_preds = status_quo_model(X)
    all_forecasts = list()
    # lots of warnings about convergence. can ignore for now.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sq_fill = 0
        linalg_fill = 0
        for index, row in X.iterrows():
            # fill in gaps
            row_interp = row.interpolate(
                method = 'linear', limit = 50,
                limit_direction = 'backward')
            # Fit ARIMA model on `lookback` most recent years data.
            # Since so much missing data exists, it is not
            # clear that including more years of interpolated
            # data is helping in terms of RMSE
            results = None
            try:
                model = sm.tsa.arima_model.ARIMA(row_interp.tolist()[-lookback:], order=order)
                results = model.fit(disp = 0)
            except (ValueError, np.linalg.linalg.LinAlgError):
                try:
                    model = sm.tsa.arima_model.ARIMA(row_interp.tolist()[-lookback:], order=backup_order)
                    results = model.fit(disp = 0)
                except (ValueError, np.linalg.linalg.LinAlgError):
                    all_forecasts.append(sq_preds.loc[index])
                    linalg_fill += 1
                    continue

            if results is None:
                all_forecasts.append(sq_preds.loc[index])
                continue

            forecasts, _, _ = results.forecast(steps=forward)
            if np.any(pd.isnull(forecasts)):
                all_forecasts.append(sq_preds.loc[index])
                sq_fill += 1
                continue
            
            all_forecasts.append(forecasts[-1])

    return(all_forecasts)

