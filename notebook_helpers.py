import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import scipy.stats

def bin_plot(x, y):
    x = np.array(x)
    bins = np.percentile(x, np.arange(0,100))
    s, edges, _ = scipy.stats.binned_statistic(x, y, statistic='mean', bins=bins)
    return sbn.scatterplot(edges[:-1] + np.diff(edges) / 2, s)

