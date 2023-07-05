"""

"""
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import matplotlib.pyplot as plt


def draw_pareto_front(data_pareto_front):
    """

    :param data_pareto_front:
    :return:
    """
    pareto_front = pd.DataFrame([[115.9, 295478.54], [110.3, 306627.52], [101.3, 290854.01]])
    # areto_front.columns = ['obj1', 'obj2']

    pareto_front.sort_values(0, inplace=True)
    pareto_front = pareto_front.values
    scores = pareto_front.to_numpy()
    x_all = scores[:, 1]
    y_all = scores[:, 2]
    x_pareto = pareto_front[:, 1]
    y_pareto = pareto_front[:, 2]

    fig = plt.figure(figsize=(6, 4))
    sns.set_style("white")

    fig = sns.scatterplot(x='obj1', y='obj2', data=pareto_front, alpha=0.7, hue='metamodel', s=200)
    plt.plot(x_pareto, y_pareto, 'o', markerfacecolor='None', markersize=13, markeredgecolor='r', markeredgewidth=2)

    plt.show()
