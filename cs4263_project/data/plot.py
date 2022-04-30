__all__ = [
    "plot",
]

import math

import matplotlib.pyplot as plt

def plot(fig, data, units=None, label_width=0, predictions=None, density=1, seperate=True, file=None, label_dates=[], labels=[]) -> None:
    """
    data: of the form of a dataframe, indexed by a datetime object, each column being a seperate series to be plotted
    units: if seperate is True, a list of strings, else a string
    label_width: label width of time series window
    predictions: of the form of a dataframe, indexed by a datetime object, each column being a seperate series to be plotted (should match up with data column names)
    density: average every 'density' number of entries to end up plotting 1/density as many entries
    file: file to save figure to (None if no save)
    """
    # Verify units
    if units==None:
        units = [""] * len(data.columns)
    elif isinstance(units, str):
        units = [units] * len(data.columns)
    elif seperate and len(units) != len(data.columns):
        print("ERROR: Make sure units is the same length as data")
        return

    # Create new data if density != 1 using the mean of rolling windows
    if density != 1:
        # data
        data = data.rolling(density).mean().iloc[::density,:]

        # predictions
        if predictions is not None:
            predictions = predictions.rolling(density).mean().loc[data.index.intersection(predictions.index),:]

    fig.patch.set_facecolor('white')

    # If plotting seperate Divide the fig into N subplots where N is the number of columns in data
    if seperate:
        ncols=math.ceil(math.sqrt(len(data.columns)))
        nrows=math.ceil( len(data.columns) / ncols)
    else: 
        plt.ylabel(units[0])
        plt.xlabel("Date")
    i = 1

    # Iterate over all columns in data
    for column, unit in zip(data.columns, units):
        # Label seperate subfigs
        if seperate:
            ax = plt.subplot(nrows,ncols,i)
            ax.set_title(column)
            plt.ylabel(unit)
            plt.xlabel('Date')
            plt.locator_params(axis='x', nbins=10)
        i+=1

        # Plot data
        plt.plot(data.index,data[column], label=column)

        # Plot labels
        if column in labels:
            plt.plot(data.loc[label_dates.intersection(data.index)][column], label=column + " labels" )

        # Plot predictions
        if label_width != 0 and column in predictions.columns:
            # plot as one long series if label width is 1
            if label_width == 1:
                plt.plot(predictions.index, predictions[column], label=column + " Predictions")

            # plot as a lot of series if label width is > 1
            else:
                pass
            
        if seperate:
            plt.legend()

    if not seperate:
        plt.legend()
    plt.tight_layout()
    plt.savefig(file)
    plt.clf()