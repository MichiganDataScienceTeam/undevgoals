import dataset

def main():

    data = dataset.UNDevGoalsDataset()
    
    # Run simple baseline model
    print('Predicting 2007 from 1972:2006')

    status_quo_rmse = data.prediction_error(error_fn='RMSE', model_name='status_quo_model',pp_fn='preprocess_avg_NANs')
    print('Status quo model RMSE:', status_quo_rmse)
    
    arima_rmse = data.prediction_error(error_fn='RMSE', model_name='arima', model_kwargs={'order': (1,1,1), 'lookback': 5},\
                                       pp_fn='preprocess_avg_NANs', pp_fn_kwargs={})
    print('ARIMA model RMSE:', arima_rmse)

if __name__ == '__main__':
    main()
