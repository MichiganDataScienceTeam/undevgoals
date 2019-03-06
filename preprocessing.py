import pandas as pd
import numpy as np

import pycountry, pycountry_convert
import json
import random
from scipy.optimize import curve_fit
import warnings

def preprocess_with_interpolation(training_set):
        """Preprecoess the data while adding in continent and region in order to better
        interpolate missing data and improve models."""
        
        X = training_set.copy()
        
        X['continent'] = ''
        
        missing = []
        
        for index, row in X.iterrows(): 
            
            country = pycountry.countries.get(name = row['Country Name'])
            
            try:
                alpha_2 = country.alpha_2
                continent = pycountry_convert.country_alpha2_to_continent_code(alpha_2)
            except(AttributeError, KeyError):
                missing.append(row['Country Name'])
            
            X.at[index, 'continent'] = continent

        missing_series = pd.Series(missing)
        missing_unique = missing_series.unique()
        
        
        for i, row in X[(X['continent'] == '')].iterrows():
            for name in missing_unique:
                
                if(row['Country Name'] == name):
                    
                    if(name == missing_unique[0]):
                        row['continent'] = 'NA'
                    
                    if(name == missing_unique[1]):
                        row['continent'] = 'SA'
                        
                    if(name == missing_unique[2]):
                        row['continent'] = 'EU'
                        
                    if(name == missing_unique[3]):
                        row['continent'] = 'AF'
                        
                    if(name == missing_unique[4]):
                        row['continent'] = 'AF'
                        
                    if(name == missing_unique[5]):
                        row['continent'] = 'AF'
                        
                    if(name == missing_unique[6]):
                        row['continent'] = 'SA'
                    
                    if(name == missing_unique[7]):
                        row['continent'] = 'EU'
                        
                    if(name == missing_unique[8]):
                        row['continent'] = 'AF'                       
                        
                    if(name == missing_unique[9]):
                        row['continent'] = 'EU'                        
                        
                    if(name == missing_unique[10]):
                        row['continent'] = 'AF'                        
                        
                    if(name == missing_unique[11]):
                        row['continent'] = 'AS'
                        
                    if(name == missing_unique[12]):
                        row['continent'] = 'AS'
                        
                    if(name == missing_unique[13]):
                        row['continent'] = 'AS'
                        
                    if(name == missing_unique[14]):
                        row['continent'] = 'AS'
                    
                    if(name == missing_unique[15]):
                        row['continent'] = 'EU'
                      
                    if(name == missing_unique[16]):
                        row['continent'] = 'AS'
                    
                    if(name == missing_unique[17]):
                        row['continent'] = 'AS'
                    
                    if(name == missing_unique[18]):
                        row['continent'] = 'AS'
                      
                    if(name == missing_unique[19]):
                        row['continent'] = 'EU'
                      
                    if(name == missing_unique[20]):
                        row['continent'] = 'OC'
                      
                    if(name == missing_unique[21]):
                        row['continent'] = 'EU'
                      
                    if(name == missing_unique[22]):
                        row['continent'] = 'NA'
                      
                    if(name == missing_unique[23]):
                        row['continent'] = 'EU'
                        
                    if(name == missing_unique[24]):
                        row['continent'] = 'NA'
                      
                    if(name == missing_unique[25]):
                        row['continent'] = 'NA'
                      
                    if(name == missing_unique[26]):
                        row['continent'] = 'NA'
                      
                    if(name == missing_unique[27]):
                        row['continent'] = 'NA'
                      
                    if(name == missing_unique[28]):
                        row['continent'] = 'AF'
                        
                    if(name == missing_unique[29]):
                        row['continent'] = 'AS'
                      
                    if(name == missing_unique[30]):
                        row['continent'] = 'SA'
                        
                    if(name == missing_unique[31]):
                        row['continent'] = 'AS'
                      
                    if(name == missing_unique[32]):
                        row['continent'] = 'NA'
                   
                    if(name == missing_unique[33]):
                        row['continent'] = 'AS'
                      
                    if(name == missing_unique[34]):
                        row['continent'] = 'AS'
                    
          
        
        return X

def preprocess_with_continent_interpolation(training_set, submit_rows_index, years_ahead=1):
    """Preprocess the training set to get the submittable training rows
    with continent-indicator-year averages filled in for missing data. These
    averages come from the ind_yr_cont_avgs.json file
    """
    X_with_cont = preprocess_with_interpolation(training_set)
    X_submit = X_with_cont.loc[submit_rows_index]

    def rename_cols(colname):
        if colname not in ['Country Name', 'Series Code', 'Series Name', 'continent']:
            return int(colname.split(' ')[0])
        else:
            return colname
    X = X_submit.rename(rename_cols, axis=1)

    with open("ind_yr_cont_avgs.json", "r") as content:
        cont_avgs = json.load(content)

    def impute_indyrcontavg(r, ind, cont):
        if pd.isna(r['value']):
            r['value'] = cont_avgs[str((ind, cont, r.name))]
            return(r)
        else:
            return(r)

    for ix,row in X.iterrows():
        ind = row['Series Code']
        cont = row['continent']
        df = row.to_frame(0)
        df.columns = ['value']
        df = df.apply(impute_indyrcontavg, axis = 1, args=(ind,cont))
        X.loc[ix] = df['value']
    # we only want the time series data for each row
    X = X.iloc[:, :-4]

    # Split prediction and target
    Y = X.iloc[:, -1]  # 2007
    X = X.iloc[:, :-1*years_ahead]  # 1972:2006 (if years_ahead==1)

    return X, Y

def preprocess_with_continent_and_linear_interpolation(training_set, submit_rows_index, years_ahead=1):
    """Preprocess the training set to get the submittable training rows
    with continent-indicator-year averages filled in for missing data. These
    averages come from the ind_yr_cont_avgs.json file
    """
    X_with_cont = preprocess_with_interpolation(training_set)
    X_submit = X_with_cont.loc[submit_rows_index]

    def func(x, a, b, c, d):
            return np.exp(c*x + d)

    def rename_cols(colname):
        if colname not in ['Country Name', 'Series Code', 'Series Name', 'continent']:
            return int(colname.split(' ')[0])
        else:
            return colname
    X = X_submit.rename(rename_cols, axis=1)

    with open("ind_yr_cont_avgs.json", "r") as content:
        cont_avgs = json.load(content)

    def impute_indyrcontavg(r, ind, cont):
        if pd.isna(r['value']):
            r['value'] = cont_avgs[str((ind, cont, r.name))]
            return(r)
        else:
            return(r)

    for ix,row in X.iterrows():
        ind = row['Series Code']
        cont = row['continent']
        df = row.to_frame(0)
        df.columns = ['value']
        df = df.apply(impute_indyrcontavg, axis = 1, args=(ind,cont))
        X.loc[ix] = df['value']
    # we only want the time series data for each row
    X = X.iloc[:, :-4]

    # Split prediction and target
    Y = X.iloc[:, -1]  # 2007
    X = X.iloc[:, :-1*years_ahead]  # 1972:2006 (if years_ahead==1)
    # Now do the linear interpolation and extrapolation part
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for ix, row in X.iterrows():
            # First interpolate
            interp_row = row.interpolate(method = 'linear')
            # Initial parameter guess, just to kick off the optimization
            guess = (0, 0, 0, 0)
            # Create copy of data to remove NaNs for curve fitting
            fit_row = interp_row.dropna()
            # Fit on non NaNs
            x = fit_row.index.astype(float).values
            y = fit_row.values
            # Curve fit series and get curve parameters
            curve_out = curve_fit(func, x, y, guess)
            # Store optimized parameters
            params = curve_out[0]
            # Impute missing values
            x = interp_row[pd.isna(interp_row)].index.astype(float).values
            interp_row[x] = [max(0,v) for v in func(x, *params)]
            X.loc[ix] = interp_row
        return X, Y

def preprocess_simple(training_set, submit_rows_index, years_ahead=1):
    """Preprocess the data for preliminary model building.

    This creates a training set where each row is a time series of a
    specific macroeconomic indicator for a specific country. The `X` table
    includes the time series from 1972 to 2006, and the 'Y' table includes
    the time series values for 2007. Missing values are coded as NaNs.

    X and Y only include rows for which we need to make submissions for the
    competition. Future iterations will include more rows to use as
    features.

    years_ahead: the number of years between data and the prediction target.

    Returns:
       X (pd.DataFrame): features for prediction
       Y (pd.Series): targets for prediction
    """
    # Select rows for prediction only
    X = training_set.loc[submit_rows_index]

    # Select and rename columns
    X = X.iloc[:, :-3]
    X = X.rename(lambda x: int(x.split(' ')[0]), axis=1)

    # Split prediction and target
    Y = X.iloc[:, -1]  # 2007
    X = X.iloc[:, :-1*years_ahead]  # 1972:2006 (if years_ahead==1)

    return X, Y

def preprocess_by_country_one_year(training_set, submit_rows_index, years_ahead=1):
    """Group data by country.

    Each row is one country, each feature is the most recent value of each
    series for that country, and each target is the next value of a target
    series.

    years_ahead: the number of years between data and the prediction target.

    Returns:
       X (pd.DataFrame): features for prediction
       Y (pd.Series): targets for prediction
    """

    # Rename columns to make indexing easier
    info_cols = training_set.iloc[:, -3:]
    training_set = training_set.iloc[:, :-3]
    training_set = training_set.rename(lambda x: int(x.split(' ')[0]), axis=1)
    training_set = pd.concat([training_set, info_cols], axis=1)

    # Mark all columns that need to be predicted in 2007
    training_set['to_predict'] = training_set.index.isin(submit_rows_index)

    # Get a list of all series codes, and all series codes that we predict
    # Not all countries will have values for all codes
    # TODO: remove some series with not many values, only keep top K
    all_series = training_set['Series Code'].value_counts()[:100]
    pred_series = training_set.loc[submit_rows_index, 'Series Code'].unique()


    # Group by country
    gb = training_set.groupby('Country Name')

    # Construct dataframe row by row
    Xrows, Yrows = [], []
    for g, group in gb:
        x = group[2007-years_ahead]
        y = group[2007]
        code = group['Series Code']
        pred = group['to_predict']

        Xrow = {}
        Yrow = {}
        for xval, yval, series, to_pred in zip(x, y, code, pred):
            if series in all_series:
                Xrow[series] = xval
            if to_pred:
                Yrow[series] = yval

        Xrow = pd.DataFrame(Xrow, index=[g])
        Yrow = pd.DataFrame(Yrow, index=[g])
        Xrows.append(Xrow)
        Yrows.append(Yrow)

    X = pd.concat(Xrows, sort=False)
    Y = pd.concat(Yrows, sort=False)

    # Impute missing values (X only)
    Xnull = pd.isnull(X)
    X = X.fillna(X.mean())

    # Remove columns that can't be imputed
    Xmeans = X.mean()
    keep_cols = Xmeans.index[~pd.isnull(Xmeans)].tolist()
    X = X[keep_cols]
    Xnull = Xnull[keep_cols]
    for col in X.columns:
        X[col+'_null'] = Xnull[col].astype(int)

    # Random split
    c = list(gb.groups.keys())
    np.random.shuffle(c)

    ctr = c[:160]
    cval = c[160:]

    Xtr = X.loc[ctr]
    Xval = X.loc[cval]
    Ytr = Y.loc[ctr]
    Yval = Y.loc[cval]

    return Xtr, Ytr, Xval, Yval

def preprocess_for_viz(training_set, submit_rows_index):
    """Preprocess the data for visualization.

    Selects rows for prediction and renames columns.
    """

    # Select rows for prediction only
    X = training_set.loc[submit_rows_index]

    # Select and rename columns
    yrs = X.iloc[:, :-3]
    names = X.iloc[:, -3:]
    yrs = yrs.rename(lambda x: int(x.split(' ')[0]), axis=1)

    df = pd.concat([yrs, names], axis=1)
    gb = df.groupby('Series Name')

    return gb

def preprocess_avg_NANs(training_set, submit_rows_index, years_ahead=1):
    """
    For NANs in most recent time period, takes average of all most recent series values with the same indicator name,
        or if there was a non NAN value in the most recent 10 years we take the most recent one  
        
    Also linearly interpolates the rest of the values in the dataframe

    Returns:
       X (pd.DataFrame): features for prediction
       Y (pd.Series): targets for prediction
    """

    # Select rows for prediction only
    full_training_rows = training_set.loc[submit_rows_index]

    # Select and rename columns
    X = full_training_rows.iloc[:, :-3]
    X = X.rename(lambda x: int(x.split(' ')[0]), axis=1)

    # Split prediction and target
    Y = X.iloc[:, -1]  # 2007
    X = X.iloc[:, :-1*years_ahead]  # 1972:2006
    
    indicators=np.unique(full_training_rows['Series Name'])
    last_column_train=X.iloc[:, -1]
    last_column_all=training_set.iloc[:,-5]
    for ind in indicators:
        
        # Find which rows in the training set and full dataset are for the indicator of interest  
        training_rows_with_indicator = last_column_train.loc[full_training_rows['Series Name'] == ind]
        all_rows_with_indicator = last_column_all.loc[training_set['Series Name'] == ind]
        
        # Find rows in training set that correspond to indicator of interest and have NAN values in most recent time period  
        NAN_training_indices_with_indicator = training_rows_with_indicator[training_rows_with_indicator.isnull()].index 
        median_of_others = np.median(all_rows_with_indicator[~all_rows_with_indicator.isnull()])
        
        # For series we need to replace NANs in, if there's a non-NAN value in the most recent 10 years we take the most recent one
        # Otherwise, we replace the value with the mean from all the time series corresponding to the same indicator
        for i in NAN_training_indices_with_indicator:
            X[X.columns[-1]][i] = median_of_others
            
            for recent_index in np.arange(2,10):
                recent_val = X[X.columns[-recent_index]][i]
                
                if not(np.isnan(recent_val)):
                    X[X.columns[-1]][i]=recent_val
                    break
    
    
    for index, row in X.iterrows():
        # Fill in gaps with linear interpolation
        row_interp = row.interpolate(
            method = 'linear', limit = 50,
            limit_direction = 'backward')
        X.loc[index]=row_interp.values
        
    return X, Y
