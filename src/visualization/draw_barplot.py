import matplotlib.pyplot as plt
import numpy as np


def draw_barplot(data, labels, xticks_labels, name):
    categories = ['m', 'c', 'i']
    values_c = data[categories].values
    if name=='a':
        colors = ['#006400', '#32CD32', '#90EE90']
        x_labels = 'Current Facilities'
    else:
        colors = ['#FFB000', '#FFC300', '#FFD700']
        x_labels = 'Proposed Facilities'
    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = np.zeros(len(labels))

    for i, category in enumerate(categories):
        ax.bar(labels, values_c[:, i], width=0.5, bottom=bottom,
               label=category, color=colors[i], edgecolor='grey')
        bottom += values_c[:, i]

    ax.set_ylim([0, 5500])
    ax.set_xlabel(x_labels, fontsize=10)
    ax.set_ylabel("Resource Stockpile Quantity", fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(xticks_labels, rotation=0)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'../results/plots/fig6_barplot_{name}.png', dpi=400)
    plt.close()
