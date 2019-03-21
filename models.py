import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.stattools import MissingDataError
import warnings
from statsmodels.tsa.api import VAR

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

def arima(X, order = (2,1,0), backup_order = (1,1,0), lookback = 8, forward=1):

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
            if forecasts[-1] > 1:
                if forward > 1:
                    one_yr_fcst = results.forecast(steps=1)[0][-1]
                    all_forecasts.append((one_yr_fcst + forecasts[-1])/2)
                else:
                    all_forecasts.append(1)
            elif forecasts[-1] < 0:
                if forward > 1:
                    one_yr_fcst = results.forecast(steps=1)[0][-1]
                    all_forecasts.append((one_yr_fcst + forecasts[-1])/2)
                else:
                    all_forecasts.append(0)
            else:
                all_forecasts.append(forecasts[-1])

    return(all_forecasts)

def var(X, lookback = 4, forward=1):
    """
    Prediction using VAR model
    """
    dat = X.iloc[:,-lookback:].values.T
    dat += 1e-3*np.random.rand(dat.shape[0],dat.shape[1])
    model = VAR(dat)
    results = model.fit()
    lag_order = results.k_ar

    return results.forecast(dat[-lag_order:], forward)[-1]

def arima_and_var(X, order = (2,1,0), backup_order = (1,1,0), arima_lookback = 4, var_lookback = 6, do_arima=False, forward = 5):
    """
    This model will do arima if the arima argument is set to True or VAR if it is set to False
    """
    if do_arima:
        out = arima(X, order = order, backup_order = backup_order, lookback = arima_lookback, forward=forward)
    else:
        out = var(X, lookback = var_lookback, forward = forward)
    return(out)


