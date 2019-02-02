import dataset
from models import status_quo_model

def main():

    data = dataset.UNDevGoalsDataset()
    X, Y = data.preprocess_simple()

    # Run simple baseline model
    print('Predicting 2007 from 1972:2006')

    status_quo_preds = status_quo_model(X)
    status_quo_rmse = data.evaluate(status_quo_preds)
    print('Status quo model RMSE:', status_quo_rmse)


if __name__ == '__main__':
    main()
