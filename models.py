import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import warnings
import numpy
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
    """Predict the most recent value using ARIMA(2,1,0) model.

    If any issues fitting time series model, use status_quo."""
    
    sq_preds = status_quo_model(X)
    X["id"] = X.index
    # get it to long form
    longX = pd.melt(X, id_vars=['id'], value_vars=list(X.columns.values)[:-1])
    # change the default column name to a proper one
    longX['year'] = longX['variable']
    longX.drop(columns = 'variable', inplace=True)
    # sort by id and year so that we can fit individual time series models for each indicator
    longX = longX.sort_values(['id', 'year'])
    # group
    grouped = longX.groupby('id')['value']
    # for each group (a country-indicator combination)
    forecasts = list()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for k, g in grouped:
            g = g[-5:]
            gi = g.interpolate(method = 'linear', limit = 50, limit_direction = 'backward')
            model = ARIMA(gi, order=(2,1,0))
            try:
                results = model.fit(disp = 0)
                forecasts.append(results.forecast()[0][0])
            except (ValueError, numpy.linalg.linalg.LinAlgError) as e:
                forecasts.append(sq_preds[k])
        return forecasts

