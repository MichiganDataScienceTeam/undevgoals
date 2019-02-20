import dataset
from preprocessing import *
from models import *
from evaluation import *

def main():

    data = dataset.UNDevGoalsDataset()
    
    print('Predicting 2007 from 1972:2006')
    
    X_simple,Y_simple = data.preprocess(pp_fn = preprocess_simple)
    status_quo_predictions_simple = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple)
    status_quo_simple_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple)
    print('Status quo model RMSE with simple preprocessing:', status_quo_simple_rmse)
    
    
    X_improved,Y_improved = data.preprocess(pp_fn = preprocess_avg_NANs)
    status_quo_predictions_improved = data.predictions(model_name=status_quo_model, preprocessed_data=X_improved)
    status_quo_improved_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_improved)
    print('Status quo model RMSE with better preprocessing:', status_quo_improved_rmse)


    arima_predictions_simple = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_simple)
    arima_rmse_simple = data.error(error_fn=RMSE, predictions=arima_predictions_simple)
    print('ARIMA model RMSE with simple preprocessing:', arima_rmse_simple)
    
    arima_predictions_improved = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_improved)
    arima_rmse_improved = data.error(error_fn=RMSE, predictions=arima_predictions_improved)
    print('ARIMA model RMSE with better preprocessing:', arima_rmse_improved)
    

if __name__ == '__main__':
    main()
