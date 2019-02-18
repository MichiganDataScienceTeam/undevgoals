import pandas as pd
import numpy as np
import os.path

import preprocessing
import evaluation
import models

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

    
    def preprocess(self, pp_fn='preprocess_simple', pp_fn_kwargs={}):
        """Preprocess data using function pp_fn (with additional kwargs if necessary) from preprocessing.py"""
        """pp_fn is string with name of preprocessing model"""
        
        return eval('preprocessing.'+pp_fn+'(self._train, self._submit_rows.index, **pp_fn_kwargs)')
    
    
    def predictions(self, model_name='status_quo_model', model_kwargs={}, pp_fn='preprocess_simple', pp_fn_kwargs={}):
        """Return predictions from model_name when fit to data preprocessed by pp_fn"""
        """pp_fn is string with name of preprocessing model, and model_name is name of prediction model"""
        """Allows input of additional keyword parameters for model and preprocessing functions"""
        
        X,Y = self.preprocess(pp_fn, **pp_fn_kwargs)
        return eval('models.'+model_name+'(X, **model_kwargs)') 
    
    
    def prediction_error(self, error_fn='RMSE', model_name='status_quo_model', model_kwargs={}, pp_fn='preprocess_simple', pp_fn_kwargs={}):
        """Check error of predictions usinf model_name and preprocessing pp_fn with error function error_fn"""
        
        X,Y = self.preprocess(pp_fn, **pp_fn_kwargs)
        return eval('evaluation.'+error_fn+'(models.'+model_name+'(X, **model_kwargs), Y)')
    

    def training_indices(self):
        """Returns list of indices that reference rows we need to predict"""
        
        X,_ = self.preprocess()
        return np.array(X.index)
    
    
    def training_indicators(self):
        """Returns list of the 8 indicators we will need to predict"""
        
        return np.unique(self._train.loc[self.training_indices()]['Series Name'])
        