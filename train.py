import dataset
from preprocessing import *
from models import *
from evaluation import *
from mlp import mlp

def main():

    data = dataset.UNDevGoalsDataset()

    # X, Y = data.preprocess(pp_fn=preprocess_by_country_all_years)

    # assert False

    # Xtr, Ytr, Xval, Yval = data.preprocess(pp_fn=preprocess_by_country_one_year)
    # preds = data.predictions(model_name=mlp, preprocessed_data=(Xtr, Ytr, Xval, Yval))
    # assert False
    #rmse = data.error(error_fn=RMSE, predictions=preds)
    #print(rmse)

    # Get all the preprocessed data
    X_simple,Y_simple = data.preprocess(pp_fn = preprocess_simple)
    X_simple_5,Y_simple_5 = data.preprocess(pp_fn = preprocess_simple, years_ahead=5)
    X_with_global_avg,Y_with_global_avg = data.preprocess(pp_fn = preprocess_avg_NANs)
    X_with_global_avg_5,Y_with_global_avg_5 = data.preprocess(pp_fn = preprocess_avg_NANs, years_ahead=5)
    X_with_cont_avg,Y_with_cont_avg = data.preprocess(pp_fn = preprocess_with_continent_interpolation)
    X_with_cont_avg_5,Y_with_cont_avg_5 = data.preprocess(pp_fn = preprocess_with_continent_interpolation, years_ahead=5)
    X_with_cont_avg_and_lin_interp, Y_with_cont_avg_and_lin_interp = data.preprocess(pp_fn = preprocess_with_continent_and_linear_interpolation)
    X_with_cont_avg_and_lin_interp_5, Y_with_cont_avg_and_lin_interp_5 = data.preprocess(pp_fn = preprocess_with_continent_and_linear_interpolation, years_ahead=5)

    # Status quo with simple preprocessing
    status_quo_predictions_simple = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple)
    status_quo_simple_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple)
    print('Status quo model RMSE with simple preprocessing (1 yr):', status_quo_simple_rmse)
    status_quo_predictions_simple_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_simple_5)
    status_quo_simple_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_simple_5)
    print('Status quo model RMSE with simple preprocessing (5 yr):', status_quo_simple_rmse_5)
    status_quo_rmse_avg = (status_quo_simple_rmse + status_quo_simple_rmse_5) / 2
    print('Status quo model RMSE with simple preprocessing (avg):', status_quo_rmse_avg)

    print()
    
    # Status quo with global avg interpolation
    status_quo_predictions_with_global_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_global_avg)
    status_quo_with_global_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_global_avg)
    print('Status quo model RMSE with global average imputation (1 yr):', status_quo_with_global_avg_rmse)
    status_quo_predictions_with_global_avg_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_global_avg_5)
    status_quo_with_global_avg_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_global_avg_5)
    print('Status quo model RMSE with global average imputation (5 yr):', status_quo_with_global_avg_rmse_5)
    status_quo_predictions_with_global_avg_rmse_avg = (status_quo_with_global_avg_rmse + status_quo_with_global_avg_rmse_5) / 2
    print('Status quo model RMSE with global average imputation (avg):', status_quo_predictions_with_global_avg_rmse_avg)

    print()

    # Status quo with continent average interpolation
    status_quo_predictions_with_cont_avg = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg)
    status_quo_with_cont_avg_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg)
    print('Status quo model RMSE with continent average imputation (1yr):', status_quo_with_cont_avg_rmse)
    status_quo_predictions_with_cont_avg_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg_5)
    status_quo_with_cont_avg_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg_5)
    print('Status quo model RMSE with continent average imputation (5 yr):', status_quo_with_cont_avg_rmse_5)
    status_quo_predictions_with_cont_avg_rmse_avg = (status_quo_with_cont_avg_rmse + status_quo_with_cont_avg_rmse_5) / 2
    print('Status quo model RMSE with continent average imputation (avg):', status_quo_predictions_with_cont_avg_rmse_avg)

    print()

    # Status quo with continent average and linear interpolation
    status_quo_predictions_with_cont_avg_lin_interp = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg_and_lin_interp)
    status_quo_with_cont_avg_lin_interp_rmse = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg_lin_interp)
    print('Status quo model RMSE with continent average imputation and linear interpolation (1yr):', status_quo_with_cont_avg_lin_interp_rmse)
    status_quo_predictions_with_cont_avg_lin_interp_5 = data.predictions(model_name=status_quo_model, preprocessed_data=X_with_cont_avg_and_lin_interp_5)
    status_quo_with_cont_avg_lin_interp_rmse_5 = data.error(error_fn=RMSE, predictions=status_quo_predictions_with_cont_avg_lin_interp_5)
    print('Status quo model RMSE with continent average imputation and linear interpolation (5 yr):', status_quo_with_cont_avg_lin_interp_rmse_5)
    status_quo_predictions_with_cont_avg_lin_interp_rmse_avg = (status_quo_with_cont_avg_lin_interp_rmse + status_quo_with_cont_avg_lin_interp_rmse_5) / 2
    print('Status quo model RMSE with continent average imputation and linear interpolation (avg):', status_quo_predictions_with_cont_avg_lin_interp_rmse_avg)

    print()

    # Arima with simple preprocessing
    arima_predictions_simple = data.predictions(model_name=arima, preprocessed_data=X_simple)
    arima_rmse_simple = data.error(error_fn=RMSE, predictions=arima_predictions_simple)
    print('ARIMA model RMSE with simple preprocessing (1 yr):', arima_rmse_simple)
    arima_predictions_simple_5 = data.predictions(model_name=arima, preprocessed_data=X_simple_5, forward=5)
    arima_rmse_simple_5 = data.error(error_fn=RMSE, predictions=arima_predictions_simple_5)
    print('ARIMA model RMSE with simple preprocessing (5 yr):', arima_rmse_simple_5)
    arima_rmse_avg = (arima_rmse_simple + arima_rmse_simple_5) / 2
    print('ARIMA model RMSE with simple preprocessing (avg):', arima_rmse_avg)

    print()

    # Arima with global avg interpolation
    arima_predictions_with_global_avg = data.predictions(model_name=arima, preprocessed_data=X_with_global_avg)
    arima_rmse_with_global_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_global_avg)
    print('ARIMA model RMSE with global average imputation (1 yr):', arima_rmse_with_global_avg)
    arima_predictions_with_global_avg_5 = data.predictions(model_name=arima, preprocessed_data=X_with_global_avg_5, forward=5)
    arima_rmse_with_global_avg_5 = data.error(error_fn=RMSE, predictions=arima_predictions_with_global_avg_5)
    print('ARIMA model RMSE with global average imputation (5 yr):', arima_rmse_with_global_avg_5)
    arima_rmse_with_global_avg_avg = (arima_rmse_with_global_avg + arima_rmse_with_global_avg_5) / 2
    print('ARIMA model RMSE with global average imputation (avg):', arima_rmse_with_global_avg_avg)

    print()
    
    # VAR with global avg interpolation
    VAR_predictions_with_global_avg = data.predictions(model_name=var, lookback=4, preprocessed_data=X_with_global_avg)
    VAR_rmse_with_global_avg = data.error(error_fn=RMSE, predictions=VAR_predictions_with_global_avg)
    print('VAR model RMSE with global average imputation (1 yr):', VAR_rmse_with_global_avg)
    VAR_predictions_with_global_avg_5 = data.predictions(model_name=var, lookback=4, preprocessed_data=X_with_global_avg_5, forward=5)
    VAR_rmse_with_global_avg_5 = data.error(error_fn=RMSE, predictions=VAR_predictions_with_global_avg_5)
    print('VAR model RMSE with global average imputation (5 yr):', VAR_rmse_with_global_avg_5)
    VAR_rmse_with_global_avg_avg = (VAR_rmse_with_global_avg + VAR_rmse_with_global_avg_5) / 2
    print('VAR model RMSE with global average imputation (avg):', VAR_rmse_with_global_avg_avg)

    print()

    # Arima with continent avg interpolation
    arima_predictions_with_cont_avg = data.predictions(model_name=arima, preprocessed_data=X_with_cont_avg)
    arima_rmse_with_cont_avg = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg)
    print('ARIMA model RMSE with continent average imputation (1 yr):', arima_rmse_with_cont_avg)
    arima_predictions_with_cont_avg_5 = data.predictions(model_name=arima, preprocessed_data=X_with_cont_avg_5, forward=5)
    arima_rmse_with_cont_avg_5 = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg_5)
    print('ARIMA model RMSE with continent average imputation (5 yr):', arima_rmse_with_cont_avg_5)
    arima_rmse_with_cont_avg_avg = (arima_rmse_with_cont_avg + arima_rmse_with_cont_avg_5) / 2
    print('ARIMA model RMSE with continent average imputation (avg):', arima_rmse_with_cont_avg_avg)

    print()

    # Arima with continent average and linear interpolation
    arima_predictions_with_cont_avg_lin_interp = data.predictions(model_name=arima, preprocessed_data=X_with_cont_avg_and_lin_interp)
    arima_with_cont_avg_lin_interp_rmse = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg_lin_interp)
    print('Arima model RMSE with continent average imputation and linear interpolation (1yr):', arima_with_cont_avg_lin_interp_rmse)
    arima_predictions_with_cont_avg_lin_interp_5 = data.predictions(model_name=arima, preprocessed_data=X_with_cont_avg_and_lin_interp_5, forward=5)
    arima_with_cont_avg_lin_interp_rmse_5 = data.error(error_fn=RMSE, predictions=arima_predictions_with_cont_avg_lin_interp_5)
    print('Arima model RMSE with continent average imputation and linear interpolation (5 yr):', arima_with_cont_avg_lin_interp_rmse_5)
    arima_predictions_with_cont_avg_lin_interp_rmse_avg = (arima_with_cont_avg_lin_interp_rmse + arima_with_cont_avg_lin_interp_rmse_5) / 2
    print('Arima model RMSE with continent average imputation and linear interpolation (avg):', arima_predictions_with_cont_avg_lin_interp_rmse_avg)

    print()


if __name__ == '__main__':
    main()
