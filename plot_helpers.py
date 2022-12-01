import numpy as np
import pdb
import matplotlib.pyplot as plt

def plot_heatmaps(X):
    ''' Visualize weights for the 10 classes in a MNIST logistic regression (colorcoded as heatmaps) '''
    if len(X.shape) == 3:   # if decentralized, plot only first worker's model
        X = X[0]

    plt.subplots(2,5, figsize=(24,10))
    for i in range(10):
        l1 = plt.subplot(2, 5, i + 1)
        l1.imshow(X[:784, i].reshape(28, 28), interpolation='nearest',cmap=plt.cm.RdBu)
        l1.set_xticks(())
        l1.set_yticks(())
        l1.set_xlabel('Class %i' % i)
    plt.suptitle('Image of the 784 weights for each 10 trained classifiers')
    plt.show()



def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """
    From: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    
    Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def plot_calibration_histogram(accuracy_bins, confidence_bins):
    x = np.arange(len(accuracy_bins))
    ax = plt.subplot()
    data = {'accuracy': accuracy_bins, 'confidence': confidence_bins}
    bar_plot(ax, data)
    plt.show()


