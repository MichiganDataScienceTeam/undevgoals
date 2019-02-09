import dataset
import models
import matplotlib.pyplot as plt
import os.path

_PLOT_DIR = './plots/'

def main():

    data = dataset.UNDevGoalsDataset()
    X, y = data.preprocess_simple()
    sq = models.status_quo_model(X)

    gb = data.preprocess_for_viz()
    #arima = models.arima(X)

    for name, group in gb:
        visualize(name, group, None)


def visualize(series_name, data, pred):
    """Visualize all series for a particular index"""

    for idx, val in data.iterrows():
        plt.plot(val.iloc[:-3])

    plt.legend(data['Country Name'])
    plt.title(series_name)

    sn = series_name.replace(' ', '_').replace('/', '_')
    fn = os.path.join(_PLOT_DIR, '%s.png'%(sn))
    plt.savefig(fn)
    plt.close()


if __name__ == '__main__':
    main()
