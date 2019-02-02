import pandas as pd
import os.path

_DATA_DIR = './data/'
_TRAINING_SET_FN = 'TrainingSet.csv'
_SUBMISSION_ROWS_FN = 'SubmissionRows.csv'

class UNDevGoalsDataset():

    def __init__(self):
        """Loads UN Development Goals dataset from disk.

        To access preprocessed version of the data, try:

        """

        training_set_fn = os.path.join(_DATA_DIR, _TRAINING_SET_FN)
        submission_rows_fn = os.path.join(_DATA_DIR, _SUBMISSION_ROWS_FN)

        self._train = pd.read_csv(training_set_fn, index_col=0)
        self._submit_rows = pd.read_csv(submission_rows_fn, index_col=0)

    def preprocess_simple(self):
        """Preprocess the data for preliminary model building.

        This creates a training set where each row is a time series of a
        specific macroeconomic indicator for a specific country. The `X` table
        includes the time series from 1972 to 2006, and the 'Y' table includes
        the time series values for 2007. Missing values are coded as NaNs.

        X and Y only include rows for which we need to make submissions for the
        competition. Future iterations will include more rows to use as
        features.

        Returns:
           X (pd.DataFrame): features for prediction
           Y (pd.Series): targets for prediction
        """

        # Select rows for prediction only
        X = self._train.loc[self._submit_rows.index]

        # Select and rename columns
        X = X.iloc[:, :-3]
        X = X.rename(lambda x: int(x.split(' ')[0]), axis=1)

        # Split prediction and target
        Y = X.iloc[:, -1]  # 2007
        X = X.iloc[:, :-1]  # 1972:2006

        return X, Y


if __name__=='__main__':

    dataset = UNDevGoalsDataset()

    X, Y = dataset.preprocess_simple()

    print(X.describe())
    print(Y.describe())
