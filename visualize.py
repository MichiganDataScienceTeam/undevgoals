import dataset
import models
import matplotlib.pyplot as plt
import os, os.path
import numpy as np
from preprocessing import *
from models import *

_PLOT_DIR = './plots_arima/'

def main():

    data = dataset.UNDevGoalsDataset()
    X, Y = data.preprocess(pp_fn = preprocess_avg_NANs)
    X5, Y5 = data.preprocess(pp_fn = preprocess_avg_NANs, years_ahead=5)

    preds = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X)
    preds5 = data.predictions(model_name=arima, order=(1,1,1), lookback=5, preprocessed_data=X5)

    preds = pd.Series(preds, index=X.index)
    preds5 = pd.Series(preds5, index=X.index)

    # This function returns a GroupBy object where each group is a development index
    gb = data.preprocess(pp_fn = preprocess_for_viz)
    for name, group in gb:
        visualize_worst(name, group, preds, preds5)


def visualize_all(series_name, data):
    """Visualize all series for a particular index.

    Each figure has way too many lines to be informative."""

    path = os.path.join(_PLOT_DIR, 'all')
    if not os.path.exists(path):
        os.mkdir(path)

    plt.figure(figsize=(30,10))
    for idx, val in data.iterrows():
        plt.plot(val.iloc[:-3])

    plt.legend(data['Country Name'])
    plt.title(series_name)

    sn = series_name.replace(' ', '_').replace('/', '_')
    fn = os.path.join(path, '%s.png'%(sn))
    plt.savefig(fn)
    print('Saved to:', fn)
    plt.close()


def visualize_worst(series_name, data, preds, preds5=None):
    """Visualize the worst predictions."""

    # Create the directory if it doesn't exist
    path = os.path.join(_PLOT_DIR, 'worst')
    if not os.path.exists(path):
        os.mkdir(path)

    # Select the predictions for this series
    if preds5 is not None:
        do5 = True
        preds5 = preds5.loc[data.index]
    else:
        do5 = False
    preds = preds.loc[data.index]

    # Compute the difference between predictions and truth
    if do5:
        diff = np.square(data[2007] - preds5)
    else:
        diff = np.square(data[2007] - preds)
    diff = diff.sort_values()

    # Only plot the 10 worst values
    to_plot = diff.iloc[:10].index
    #to_plot = diff.iloc[-10:].index
    data = data.loc[to_plot]
    preds = preds.loc[to_plot]
    if do5:
        preds5 = preds5.loc[to_plot]

    # Build the figure
    plt.figure(figsize=(15,5))
    cmap = plt.cm.Set1(np.linspace(0, 1, 10))
    for (idx, val), color in zip(data.iterrows(), cmap):
        notnull = val.iloc[:-3].notnull()
        # Dashed line: interpolated values
        plt.plot(val.iloc[:-3].loc[notnull], color=color, ls='--', label='')
        # Dotted line: predictions
        plt.plot([2006, 2007], [val.iloc[-5], preds.loc[idx]], ls=':', color=color, label='')
        if do5:
            plt.plot([2002, 2007], [val.iloc[-9], preds5.loc[idx]], ls=':', color=color, label='')
        # Solid line: actual values
        plt.plot(val.iloc[:-3], color=color, label=val['Country Name'])

    plt.legend()
    plt.title(series_name)

    # Save
    sn = series_name.replace(' ', '_').replace('/', '_')
    fn = os.path.join(path, '%s.png'%(sn))
    plt.savefig(fn)
    print('Saved to:', fn)
    plt.close()


if __name__ == '__main__':
    main()
