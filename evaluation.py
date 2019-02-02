import pandas as pd
import numpy as np

import argparse
import dataset

def RMSE(predictions, labels):
    """Computes the Root Mean Squared Error (RMSE) between two pd.Series."""
    return np.sqrt(np.mean(np.square(predictions - labels)))
