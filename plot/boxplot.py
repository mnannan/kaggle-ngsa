import matplotlib.pyplot as plt
import pandas as pd


def box_plot_feature(series: pd.Series, positive: pd.Series, negative: pd.Series, title: str=None):
    """
    This util allow us to quickly series repartition for positive and negative inputs
    :param series: pandas serie we want to plot
    :param positive: Mask of positive inputs (e.g df.category == 1)
    :param negative: Mask of negative inputs (e.g df.category == 0)
    :param title: title displayed
    :return:
    """
    positive = series[positive]
    negative = series[negative]

    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axarr[0].boxplot(positive)
    axarr[1].boxplot(negative)
    if title:
        axarr[0].set_title(f'{title} when connected')
        axarr[1].set_title(f'{title} when not connected')
