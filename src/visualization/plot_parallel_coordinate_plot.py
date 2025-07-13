import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker


def plot_parallel_coordinate_plot(data_co_sorted, data_MeanRT_sorted, data_cost_sorted):
    fig, ax = plt.subplots(figsize=(2,3))
    kwargs = {'lw':0.6} #'alpha':1,
    ax = pd.plotting.parallel_coordinates(data_co_sorted, class_column='coverage_percentage',
                                     cols=['w1','w2', 'w3'],
                                     colormap=plt.get_cmap('hot'),
                                    axvlines=False,
                                     **kwargs                      )
    plt.yscale("log")
    plt.ylabel('Coverage', fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xticklabels(['$w_1$', '$w_2$', '$w_3$'])
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1,1), fontsize = 8)
    ax.grid(False)
    fig.savefig(f'../results/plots/Fig8a PCP.png', transparent=False, dpi=400, bbox_inches = "tight")

    fig, ax = plt.subplots(figsize=(2,3))
    kwargs = {'lw':0.6} #'alpha':1,
    ax = pd.plotting.parallel_coordinates(data_MeanRT_sorted, class_column='MeanResponseTime',
                                     cols=['w1','w2', 'w3'],
                                     colormap=plt.get_cmap('hot'),axvlines=False,
                                     **kwargs                      )
    plt.yscale("log")
    plt.ylabel('Mean Response Time', fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xticklabels(['$w_1$', '$w_2$', '$w_3$'])
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1,1), fontsize = 8)
    ax.grid(False)
    fig.savefig(f'../results/plots/Fig8b PCP.png', transparent=False, dpi=400, bbox_inches = "tight")

    fig, ax = plt.subplots(figsize=(2,3))
    kwargs = {'lw':0.6} #'alpha':1,
    ax = pd.plotting.parallel_coordinates(data_cost_sorted, class_column='objValues2',
                                     cols=['w1','w2', 'w3'],
                                     colormap=plt.get_cmap('hot'),axvlines=False,
                                     **kwargs                      )
    plt.yscale("log")
    plt.ylabel('Objective 2', fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xticklabels(['$w_1$', '$w_2$', '$w_3$'])
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1,1), fontsize = 7)
    ax.grid(False)
    fig.savefig(f'../results/plots/Fig8c PCP.png', transparent=False, dpi=400, bbox_inches = "tight")