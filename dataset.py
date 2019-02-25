import pandas as pd
import numpy as np
import os.path

from preprocessing import *
from evaluation import *
from models import *

_DATA_DIR = './data/'
_TRAINING_SET_FN = 'TrainingSet.csv'
_SUBMISSION_ROWS_FN = 'SubmissionRows.csv'


class UNDevGoalsDataset():

    def __init__(self):
        """Loads UN Development Goals dataset from disk"""

        training_set_fn = os.path.join(_DATA_DIR, _TRAINING_SET_FN)
        submission_rows_fn = os.path.join(_DATA_DIR, _SUBMISSION_ROWS_FN)

        self._train = pd.read_csv(training_set_fn, index_col=0)
        self._submit_rows = pd.read_csv(submission_rows_fn, index_col=0)


    def preprocess(self, pp_fn=preprocess_simple, **pp_fn_kwargs):
        """
        Preprocess data using function pp_fn (with additional kwargs if necessary) from preprocessing.py

        Args:
            pp_fn: Name of preprocessing function from preprocessing.py
            pp_fn_kwargs: Keyword arguments for preprocessing function

        Returns:
            Output of preprocessing function applied to training data restricted to rows of interest
        """

        return pp_fn(self._train, self._submit_rows.index, **pp_fn_kwargs)


    def predictions(self, preprocessed_data, model_name=status_quo_model, **model_kwargs):
        """Return predictions from model_name given preprocessed data

        Args:
            model_name: Name of prediction model from models.py
            model_kwargs: Model function keyword arguments
            preprocessed_data: Data formatted to be passed in for predictions

        Good idea to have option for returning pickled representation of model when defining the functions in models.py.
        This option would go into the model_kwargs argument here.

        Returns:
            Predictions for test column using this model and the passed in data
        """

        return model_name(preprocessed_data, **model_kwargs)


    def error(self, predictions, error_fn=RMSE, **error_fn_kwargs):
        """
        Check error of predictions with error function error_fn

        Args:
            error_fn: Name of error function from evaluation.py
            error_fn_kwargs: Keyword arguments for error function
            predictions: Predicted test column values


        Returns:
            Error on predictions based on true values Y

        """

        _, Y = self.preprocess()
        return error_fn(predictions, Y, **error_fn_kwargs)


    def training_indices(self):
        """Returns list of indices that reference rows we need to predict"""

        X, _ = self.preprocess()
        return np.array(X.index)


    def training_indicators(self):
        """Returns list of the 8 indicators we will need to predict"""

        return np.unique(self._train.loc[self.training_indices()]['Series Name'])