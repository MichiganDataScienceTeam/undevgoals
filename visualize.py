import dataset
import models
import matplotlib.pyplot as plt
import os, os.path
import numpy as np

_PLOT_DIR = './plots/'

def main():

    data = dataset.UNDevGoalsDataset()
    X, y = data.preprocess_simple()
    sq = models.status_quo_model(X)

    # Currently the arima model doesn't output a DataFrame
    #arima = models.arima(X)

    # This function returns a GroupBy object where each group is a development index
    gb = data.preprocess_for_viz()
    for name, group in gb:
        visualize_all(name, group)
        visualize_worst(name, group, sq)


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


def visualize_worst(series_name, data, preds):
    """Visualize the worst predictions."""

    # Create the directory if it doesn't exist
    path = os.path.join(_PLOT_DIR, 'worst')
    if not os.path.exists(path):
        os.mkdir(path)

    # Select the predictions for this series
    preds = preds.loc[data.index]

    # Compute the difference between predictions and truth
    diff = np.square(data[2007] - preds)
    diff = diff.sort_values()

    # Only plot the 10 worst values
    to_plot = diff.iloc[-10:].index
    data = data.loc[to_plot]
    preds = preds.loc[to_plot]

    # Build the figure
    plt.figure(figsize=(15,5))
    cmap = plt.cm.Set1(np.linspace(0, 1, 10))
    for (idx, val), color in zip(data.iterrows(), cmap):
        notnull = val.iloc[:-3].notnull()
        # Dashed line: interpolated values
        plt.plot(val.iloc[:-3].loc[notnull], color=color, ls='--', label='')
        # Dotted line: predictions
        plt.plot([2006, 2007], [val.iloc[-5], preds.loc[idx]], ls=':', color=color, label='')
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
