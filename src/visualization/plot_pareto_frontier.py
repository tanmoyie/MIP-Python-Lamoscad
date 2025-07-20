import numpy as np
import matplotlib.pyplot as plt


def plot_pareto_frontier(df):
    df_unique = df.drop_duplicates(subset=["Max Coverage", "Min Cost"])
    flat_list_max_coverage = df_unique["Max Coverage"].tolist()
    flat_list_min_cost = df_unique["Min Cost"].tolist()

    sorted_indices = np.argsort(-np.array(flat_list_max_coverage))
    pareto_frontier = []
    current_min_cost = np.inf

    for idx in sorted_indices:
        if flat_list_min_cost[idx] < current_min_cost:
            pareto_frontier.append((flat_list_max_coverage[idx], flat_list_min_cost[idx]))
            current_min_cost = flat_list_min_cost[idx]

    pareto_frontier = np.array(pareto_frontier)
    dominated_points = np.array([
        (c, co) for c, co in zip(flat_list_max_coverage, flat_list_min_cost)
        if (c, co) not in pareto_frontier
    ])

    # Plotting
    plt.figure(figsize=(5, 4))
    if len(dominated_points) > 0:
        plt.scatter(dominated_points[:, 0], dominated_points[:, 1], color='green', label='Feasible Dominated',
                    alpha=0.7, s=100)
    plt.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], color='red', label='Non-dominated', edgecolors='black',
                s=100)
    plt.plot(pareto_frontier[:, 0], pareto_frontier[:, 1], 'r--', label='Pareto Frontier', linewidth=2)

    plt.xlabel('Max Coverage objective', fontsize=11, fontname='Times New Roman')
    plt.ylabel('Min Cost objective', fontsize=11, fontname='Times New Roman')
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('../results/plots/fig11_pareto_frontier.png', dpi=400)
    plt.show()
    # plt.close()
