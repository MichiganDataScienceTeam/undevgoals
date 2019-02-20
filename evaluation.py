import numpy as np

def RMSE(predictions, labels):
    """Computes the Root Mean Squared Error (RMSE) between two pd.Series."""
    return np.sqrt(np.mean(np.square(predictions - labels)))
