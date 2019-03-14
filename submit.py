import dataset
import pandas as pd
import numpy as np
from preprocessing import *
from models import *
import sys

def main():

	num_args = len(sys.argv)
	if num_args > 1:
		if sys.argv[1] == 'arima_global_avg':
			data = dataset.UNDevGoalsDataset()
			X_submit = data.preprocess(pp_fn = preprocess_for_submission_with_global_avg_and_lin_interp)
			arima_predictions_with_global_avg_lin_interp = data.predictions(model_name=arima, preprocessed_data=X_submit, lookback = 4)
			arima_predictions_with_global_avg_lin_interp_5 = data.predictions(model_name=arima, preprocessed_data=X_submit, lookback = 4,  forward=5)
			one_yr_output = pd.Series(arima_predictions_with_global_avg_lin_interp, index = X_submit.index, name = '2008 [YR2008]')
			five_yr_output = pd.Series(arima_predictions_with_global_avg_lin_interp_5, index = X_submit.index, name = '2012 [YR2012]')
			out_df = pd.concat([one_yr_output, five_yr_output], axis=1)
			out_df.to_csv('SubmissionRows.csv')
		elif sys.argv[1] == 'arima_continent_avg':
			data = dataset.UNDevGoalsDataset()
			X_submit = data.preprocess(pp_fn = preprocess_for_submission_with_cont_avg_and_lin_interp)
			arima_predictions_with_cont_avg_lin_interp = data.predictions(model_name=arima, preprocessed_data=X_submit, lookback = 6)
			arima_predictions_with_cont_avg_lin_interp_5 = data.predictions(model_name=arima, preprocessed_data=X_submit, lookback = 6, forward=5)
			one_yr_output = pd.Series(arima_predictions_with_cont_avg_lin_interp, index = X_submit.index, name = '2008 [YR2008]')
			five_yr_output = pd.Series(arima_predictions_with_cont_avg_lin_interp_5, index = X_submit.index, name = '2012 [YR2012]')
			out_df = pd.concat([one_yr_output, five_yr_output], axis=1)
			out_df.to_csv('SubmissionRows.csv')
		else:
			raise Exception("You have to submit a valid type of submittable as a command line argument")



if __name__ == '__main__':
	main()