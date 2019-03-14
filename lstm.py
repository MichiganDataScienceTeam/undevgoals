import pandas as pd
import numpy as np
# import statsmodels as sm
# from statsmodels.tsa.stattools import MissingDataError
# import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

def normdata (data):

  scaler = StandardScaler()
  array = []
  nations = []

  for nation in data:
    nations.append(nation)
    array.append(data[nation].values.astype(float))
  array = np.array(array)

  num_nations = array.shape[0]
  num_years = array.shape[1]
  num_series = array.shape[2]

  print("norm...data.....")
  array = np.reshape(array,(num_nations*num_years,num_series))
  array = scaler.fit_transform(array)
  array = np.reshape(array,(num_nations,num_years,num_series))
  print("norm...data.....finished")

  data_norm = {}
  i = 0
  for nation in nations:
    x = pd.DataFrame(array[i,:,:], index = data[nation].index, columns = data[nation].columns)
    data_norm[nation] = x
    i = i + 1

  return data_norm, scaler

def create_array(data, mask_X, mask_Y, nations,  lookback, xmasked = False):

  X = []
  Y =[]
  Y_mask = []

  for nation in nations:
    print (nation)
    array = data[nation].values
    array_mask_X = mask_X[nation].values
    num_years = array.shape[0]

    if num_years < lookback:
      print ("ERROR: dimension of data is wrong in create_array")
      break

    # check if for forcast
    if num_years == lookback:
      x = array
      if xmasked:
        x = np.append(x,array_mask_X, axis = 1)
      X.append(x)
      continue

    array_mask_Y = mask_Y[nation].values
    # for train and test
    # construct 3D array from 2D arrays
    for start in range(0,num_years-lookback):
      # print (start)
      x = array[start:start+lookback,:]
      x_mask = array_mask_X[start:start+lookback,:]
      if xmasked:
        x = np.append(x, x_mask, axis = 1)
      X.append(x)
      y = array[start+lookback, :]
      Y.append(y)
      y_mask = array_mask_Y[start+lookback, :]
      Y_mask.append(y_mask)
  # print("converting")
  X = np.array(X)
  # print ("X Shape: ", X.shape)
  Y = np.array(Y)
  # print ("Y Shape: ", Y.shape)
  Y_mask = np.array(Y_mask)
  #Y_mask = Y.reshape(Y_mask.shape[0],1,Y_mask.shape[1])
  # print ("Y mask: ", Y_mask.shape)
  return X, Y, Y_mask


def lstm(preprocessed_data, lookback = 5):
  # credit to https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
  # Set seed to be able to reproduce

  data, pred_cols, sub_rows = preprocessed_data

  # normalize the dataset (mean=0 std=1 for each series in all years and all coutries)

  norm, scaler = normdata(data)

  # mask, intepolate data and fill remaining nan with 0s
  mask_Y = {} # mask Y is used for evaluation of loss function
  mask_X = {} # mask X is used as a feature for X values
  clean = {} # this is the data after cleaning
  nations = [] # the list of nations to keep track on the order

  for nation in norm:
    nations.append(nation)

    # mask first as mask_Y
    mask_Y[nation] = pd.notnull(norm[nation]).astype(int)
    # Do the intepolation
    clean[nation] = norm[nation].interpolate(method = 'linear', axis = 0,  limit_direction = 'both')
    # mask again 
    # 0 for those NAN cannot interpolate (missing the whole series for that nation)
    # 1 for those NAN can interpolate
    # 2 for those are not NAN
    mask_X[nation] = pd.notnull(clean[nation]).astype(int)
    mask_X[nation] = mask_X[nation].add(mask_Y[nation])
    # fill 0s (average) for those nan
    clean[nation] = clean[nation].fillna(0)

  # also do the normalization for the mask
  mask_norm, mask_scaler = normdata(mask_X) 

  # split the dataset for train, test and forecast
  train = {}
  train_mask_X = {}
  train_mask_Y = {}

  test = {}
  test_mask_X = {}
  test_mask_Y = {}

  forecast = {} 
  forecast_mask_X = {}

  print("spliting.....")

  for nation in nations:
    # training is all the dataset except the last year
    train[nation] = clean[nation].iloc[:-1,:]
    train_mask_X[nation] = mask_norm[nation].iloc[:-1,:]
    train_mask_Y[nation] = mask_Y[nation].iloc[:-1,:]

    # testing is the data for the last year and the previous 'look back' years
    year = lookback + 1
    test[nation] = clean[nation].iloc[-year:,:]
    test_mask_X[nation] = mask_norm[nation].iloc[-year:,:]
    test_mask_Y[nation] = mask_Y[nation].iloc[-year:,:]

    # forcast is the last 'look back' years
    forecast[nation] = clean[nation].iloc[-lookback:,:]
    forecast_mask_X[nation] = mask_norm[nation].iloc[-lookback:,:]

  # convert to np.array for each of them
  dummy = {}

  print("creating array....train")
  trainX, trainY, trainY_mask = create_array(train, train_mask_X, train_mask_Y, nations, lookback)
  print("creating array....test")
  testX, testY, testY_mask = create_array(test, test_mask_X, test_mask_Y, nations, lookback)
  print("creating array....forecast")
  forecastX, dummy, dummy_mask = create_array(forecast, forecast_mask_X, dummy, nations, lookback)

  # Check shape
  print("trainX: ", trainX.shape, "; trainY: ", trainY.shape, "; trainY_mask: ", trainY_mask.shape)
  print("testX: ", testX.shape, "; testY: ", testY.shape, "; testY_mask: ", testY_mask.shape)
  print("forecastX: ", forecastX.shape)

  # Build the model
  #################
  ncell = 50
  nepochs = 500
  #################
  num_nations = len(nations)
  print("n_nations: ", num_nations)
  num_batch = int(trainX.shape[0]/num_nations)
  print("num_batch", num_batch)
  num_years_input = lookback
  num_features_X = trainX.shape[2]
  num_features_Y = trainY.shape[1]

  model = Sequential()
  model.add(LSTM(ncell, input_shape = (num_years_input,num_features_X)))
  model.add(Dense(num_features_Y))
  model.compile(loss='mean_squared_error', optimizer = 'RMSprop')
  history = model.fit(trainX, trainY, epochs=nepochs, batch_size=num_batch, 
    validation_data=(testX, testY), verbose=2, shuffle=False)
  # plot history
  plt.plot(history.history['loss'], label='train error')
  plt.plot(history.history['val_loss'], label='test error')
  plt.title("Prediction")
  plt.legend()
  plt.xlabel("nepochs")
  plt.ylabel("Loss")
  plt.savefig("lstm.png")
  plt.close()

  ###########
  trainY_pred = model.predict(trainX)
  testY_pred = model.predict(testX)
  forecastY_pred = model.predict(forecastX)

  #### inverse
  trainY_pred = np.multiply(scaler.inverse_transform(trainY_pred), trainY_mask)
  testY_pred = np.multiply(scaler.inverse_transform(testY_pred),testY_mask)
  trainY = np.multiply(scaler.inverse_transform(trainY),trainY_mask)
  testY = np.multiply( scaler.inverse_transform(testY),testY_mask)
  forecastY = scaler.inverse_transform(forecastY_pred)

  ####
  err_train = math.sqrt(mean_squared_error(trainY,trainY_pred))
  print("######Errors with all series###########")
  print("training: ", err_train)
  err_test = math.sqrt(mean_squared_error(testY,testY_pred))
  print("testing: ", err_test)

  ####
  # print("RMSE with submitted rows")
  # nation0 = nations[0]
  # df_testY_pred = pd.DataFrame(testY_pred,index=nations, columns = test[nation].columns)
  # df_testY = pd.DataFrame(testY,index=nations, columns = test[nation].columns)
  # print(df_testY_pred.head())

  # df_testY_pred_sub =  df_testY_pred.lookup(sub_rows['Country Name'], sub_rows['Series Code'])
  # print(df_testY_pred_sub)
  # testY_pred_sub = []
  # testY_sub = []
  # for index, row in sub_rows.iterrows():
  #   row_nation = row.values[0]
  #   row_series = row.values[1]
  #   print(row_nation)
  #   print(row_series)
  #   sub = df_testY_pred.loc(row_nation,row_series)
  #   print(sub)
  #   testY_pred_sub.append(sub)
  #   testY_sub.append(df_testY.loc(row_nation,row_series).values)

  err_test_sub = math.sqrt(mean_squared_error(testY_sub,testY_pred_sub))
  print("testing: ", err_test_sub)





  ### print errors



  return
    




