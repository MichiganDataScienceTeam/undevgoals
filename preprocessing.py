import pandas as pd
import numpy as np

import pycountry, pycountry_convert

def preprocess_with_interpolation(training_set):
        """Preprecoess the data while adding in continent and region in order to better
        interpolate missing data and improve models."""
        
        X = training_set._train
        
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

def preprocess_simple(training_set, submit_rows_index):
    """Preprocess the data for preliminary model building.

    This creates a training set where each row is a time series of a
    specific macroeconomic indicator for a specific country. The `X` table
    includes the time series from 1972 to 2006, and the 'Y' table includes
    the time series values for 2007. Missing values are coded as NaNs.

    X and Y only include rows for which we need to make submissions for the
    competition. Future iterations will include more rows to use as
    features.

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
    X = X.iloc[:, :-1]  # 1972:2006

    return X, Y

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

def preprocess_avg_NANs(training_set, submit_rows_index):
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
    X = X.iloc[:, :-1]  # 1972:2006
    
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
