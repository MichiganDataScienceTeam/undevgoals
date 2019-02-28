import dataset
from preprocessing import *
from models import *
from evaluation import *

def main():

    data = dataset.UNDevGoalsDataset()

    X_simple,Y_simple = data.preprocess(pp_fn = preprocess_simple)
    X_simple_5,Y_simple_5 = data.preprocess(pp_fn = preprocess_simple, years_ahead=5)
    X_with_global_avg,Y_with_global_avg = data.preprocess(pp_fn = preprocess_avg_NANs)
    X_with_global_avg_5,Y_with_global_avg_5 = data.preprocess(pp_fn = preprocess_avg_NANs, years_ahead=5)
    X_with_cont_avg,Y_with_cont_avg = data.preprocess(pp_fn = preprocess_with_continent_interpolation)
    X_with_cont_avg_5,Y_with_cont_avg_5 = data.preprocess(pp_fn = preprocess_with_continent_interpolation, years_ahead=5)

    status_quo_predictions_simple = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple)
    status_quo_simple_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple)
    print('Status quo model RMSE with simple preprocessing (1 yr):', status_quo_simple_rmse)


    status_quo_predictions_simple_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple_5)
    status_quo_simple_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple_5)
    print('Status quo model RMSE with simple preprocessing (5 yr):', status_quo_simple_rmse_5)

    status_quo_rmse_avg = (status_quo_simple_rmse + status_quo_simple_rmse_5) / 2
    print('Status quo model RMSE with simple preprocessing (avg):', status_quo_rmse_avg)

    print()

    status_quo_predictions_with_global_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_global_avg)
    status_quo_with_global_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_global_avg)
    print('Status quo model RMSE with global average imputation:', status_quo_with_global_avg_rmse)

    status_quo_predictions_with_cont_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg)
    status_quo_with_cont_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg)
    print('Status quo model RMSE with continent average imputation:', status_quo_with_cont_avg_rmse)

    print()

    X_improved,Y_improved = data.preprocess(pp_fn = preprocess_avg_NANs)
    status_quo_predictions_improved = data.predictions(model_name=status_quo_model, preprocessed_data=X_improved)
    status_quo_improved_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_improved)
    print('Status quo model RMSE with better preprocessing (1 yr):', status_quo_improved_rmse)

    X_improved_5,Y_improved_5 = data.preprocess(pp_fn = preprocess_avg_NANs, years_ahead=5)
    status_quo_predictions_improved_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_improved_5)
    status_quo_improved_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_improved_5)
    print('Status quo model RMSE with better preprocessing (5 yr):', status_quo_improved_rmse_5)

    status_quo_improved_rmse_avg = (status_quo_improved_rmse + status_quo_improved_rmse_5) / 2
    print('Status quo model RMSE with better preprocessing (avg):', status_quo_improved_rmse_avg)

    print()

    arima_predictions_simple = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_simple)
    arima_rmse_simple = data.error(error_fn=RMSE, predictions=arima_predictions_simple)
    print('ARIMA model RMSE with simple preprocessing (1 yr):', arima_rmse_simple)

    arima_predictions_simple_5 = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_simple_5)
    arima_rmse_simple_5 = data.error(error_fn=RMSE, predictions=arima_predictions_simple_5)
    print('ARIMA model RMSE with simple preprocessing (5 yr):', arima_rmse_simple_5)

    arima_rmse_avg = (arima_rmse_simple + arima_rmse_simple_5) / 2
    print('ARIMA model RMSE with simple preprocessing (avg):', arima_rmse_avg)

    print()

    arima_predictions_with_global_avg = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_global_avg)
    arima_rmse_with_global_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_global_avg)
    print('ARIMA model RMSE with global average imputation (1 yr):', arima_rmse_with_global_avg)

    arima_predictions_with_global_avg_5 = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_global_avg_5)
    arima_rmse_with_global_avg_5 = data.error(error_fn=RMSE, predictions=arima_predictions_with_global_avg_5)
    print('ARIMA model RMSE with global average imputation (5 yr):', arima_rmse_with_global_avg_5)

    arima_rmse_with_global_avg_avg = (arima_rmse_with_global_avg + arima_rmse_with_global_avg_5) / 2
    print('ARIMA model RMSE with global average imputation (avg):', arima_rmse_with_global_avg_avg)

    print()

    arima_predictions_with_cont_avg = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_cont_avg)
    arima_rmse_with_cont_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg)
    print('ARIMA model RMSE with continent average imputation (1 yr):', arima_rmse_with_cont_avg)

    arima_predictions_with_cont_avg_5 = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X_with_cont_avg_5)
    arima_rmse_with_cont_avg_5 = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg_5)
    print('ARIMA model RMSE with continent average imputation (5 yr):', arima_rmse_with_cont_avg_5)

    arima_rmse_with_cont_avg_avg = (arima_rmse_with_cont_avg + arima_rmse_with_cont_avg_5) / 2
    print('ARIMA model RMSE with continent average imputation (avg):', arima_rmse_with_cont_avg_avg)
    

if __name__ == '__main__':
    main()
