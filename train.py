import dataset
from preprocessing import *
from models import *
from evaluation import *

def main():

    data = dataset.UNDevGoalsDataset()
    
    print('Predicting 2007 from 1972:2006')
    
    X_simple,Y_simple = data.preprocess(pp_fn = preprocess_simple)
    X_with_global_avg,Y_with_global_avg = data.preprocess(pp_fn = preprocess_avg_NANs)
    X_with_cont_avg,Y_with_cont_avg = data.preprocess(pp_fn = preprocess_with_continent_interpolation)


    status_quo_predictions_simple = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple)
    status_quo_simple_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple)
    print('Status quo model RMSE with simple preprocessing:', status_quo_simple_rmse)
       
    
    status_quo_predictions_with_global_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_global_avg)
    status_quo_with_global_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_global_avg)
    print('Status quo model RMSE with global average imputation:', status_quo_with_global_avg_rmse)


    status_quo_predictions_with_cont_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg)
    status_quo_with_cont_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg)
    print('Status quo model RMSE with continent average imputation:', status_quo_with_cont_avg_rmse)


    arima_predictions_simple = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_simple)
    arima_rmse_simple = data.error(error_fn=RMSE, predictions=arima_predictions_simple)
    print('ARIMA model RMSE with simple preprocessing:', arima_rmse_simple)

    
    arima_predictions_with_global_avg = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_global_avg)
    arima_rmse_with_global_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_global_avg)
    print('ARIMA model RMSE with global average imputation:', arima_rmse_with_global_avg)


    arima_predictions_with_cont_avg = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_cont_avg)
    arima_rmse_with_cont_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg)
    print('ARIMA model RMSE with continent average imputation:', arima_rmse_with_cont_avg)
    

if __name__ == '__main__':
    main()
