# undevgoals
United Nations Millennium Development Goals Competition: https://www.drivendata.org/competitions/1/united-nations-millennium-development-goals/

## Setup

1. Install `virtualenv` if you haven't already.

```
pip install --upgrade virtualenv
```

2. Create a new `virtualenv` with the correct dependencies:

```
virtualenv undg -p python3
source undg/bin/activate
pip install -r requirements.txt
```

3. Place data in `./data/`. There should be two files:

 - TrainingSet.csv
 - SubmissionRows.csv

4. Run the training code

```
python train.py
```

5. To test new preprocessing techniques, write your function into `preprocessing.py`, making sure it is of the same format as the other functions in the file. For new prediction models and error evaluation functions, make corresponding changes to `models.py` or `evaluation.py`.

6. Check the dataset.py file for information about methods for the UNDevGoalsDataset() class.  

## Welcome to the UN Dev Goals Project!

_Thanks to Jacsarge for this intro_

If you've reached this point, you have jupyter up and running on your device which is great!
Pat yourself on the back!

Now you might be asking yourself,

### What's the point of this project?

#### From the project website:

The UN measures progress towards these goals using indicators such as percent of the population making over one dollar per day. Your task is to predict the change in these indicators one year and five years into the future. Predicting future progress will help us to understand how we achieve these goals by uncovering complex relations between these goals and other economic indicators. The UN set 2015 as the target for measurable progress. Given the data from 1972 - 2007, you need to predict a specific indicator for each of these goals in 2008 and 2012.

### Notes on the Data:

* Time Series Data
* Uneven distribution of data e.g. only one row for 'combating malaria'
* Lots of missing values
* Size of original data: 195402
* Size of data to be predicted on: 737

#### TrainingSet.csv and SubmissionRows.csv

#### Training Set

This is all of the data that the competition provides. We will use this to predict the values in submission rows for the years 2008 and 2012.

#### Submission Rows

These are the individual rows from Training Set that we need to predict for the competition. It is broken up by year. These rows are given as the competition believes they provide enough data for us to predict accurately for 2008 and 2012.

#### Files

* dataset.py
    * This preprocesses the data given so that we can use it to make predictions. Move all preprocessing code to the dataset class when you can.
* evaluation.py
    * This calculates the Root Mean Squared Error (RMSE) for pd.Series passed into it.
* models.py
    * This file currently holds the status_quo and arima models we are using to make our predictions.
    * status_quo
        * The status_quo model assumes that everything continues as it had the last year and translates that value over into our prediction.
    * arima
        * ARIMA stands for Autoregressive Integrated Moving Average. This model is fitted to time series data and is one of the models we will use to make our predictions. If the ARIMA prediction is worse than the status_quo prediction then it uses that instead
* requirements.txt
    * This text document will be used to install the required modules to your virtualenv for this project
* train.py
    * This file trains our current models on a set of preprocessed data
* visualize.py
    * This file visualizes all series for a particular index
