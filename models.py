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

def arima(X):
    """Predict the most recent value using ARIMA(1,1,1) model.

    If any issues fitting time series model, use status_quo."""
    
    sq_preds = status_quo_model(X)
    forecasts = list()
    # lots of warnings about convergence. can ignore for now.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for index, row in X.iterrows():
            # fill in gaps
            row_interp = row.interpolate(
                method = 'linear', limit = 50,
                limit_direction = 'backward')
            # fit ARIMA model on 5 most recent years data
            # since so much missing data exists, it is not
            # clear that including more years of interpolated
            # data is helping in terms of RMSE
            model = sm.tsa.arima_model.ARIMA(row_interp.tolist()[-5:], order=(1,1,1))
            try:
                results = model.fit(disp = 0)
                if pd.isnull(results.forecast()[0][0]) or np.abs(results.forecast()[0][0])>2:
                    forecasts.append(sq_preds.loc[index])
                else: 
                    forecasts.append(results.forecast()[0][0])
            except (ValueError, np.linalg.linalg.LinAlgError, MissingDataError) as e:
                    forecasts.append(sq_preds.loc[index])
    return(forecasts)
