
def draw_parallel_coordinate_plot(df_fig8):
    data_coverage_sorted = df_fig8.sort_values(by='Coverage_%', ascending=False)
    data_MeanResponseTime_sorted = df_fig8.sort_values(by='Mean_Response_Time', ascending=False)
    data_costObj2_sorted = df_fig8.sort_values(by='Objective_2', ascending=False)

    # Corrected dataset info with proper column names
    datasets = [
        (data_coverage_sorted, 'Coverage_%', 'Coverage'),
        (data_MeanResponseTime_sorted, 'Mean_Response_Time', 'Mean Response Time'),
        (data_costObj2_sorted, 'Objective_2', 'Objective 2')
    ]
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Define a function to compute 10 evenly spaced legend labels
    def get_evenly_spaced_legend_items(data_column, num_items=10):
        values = data_column.values
        min_val, max_val = np.nanmin(values), np.nanmax(values)
        thresholds = np.linspace(min_val, max_val, num_items)
        return thresholds

    import matplotlib.patches as patches
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))

    for ax, (data, class_column, ylabel) in zip(axes, datasets):
        legend_vals = get_evenly_spaced_legend_items(data[class_column])[::-1]  # Descending

        pd.plotting.parallel_coordinates(
            data, class_column=class_column, cols=['w1', 'w2', 'w3'], axvlines=False,
            colormap=plt.get_cmap('hot'), lw=0.6, ax=ax
        )

        # Remove vertical lines
        for line in ax.lines:
            line.set_zorder(1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['$w_1$', '$w_2$', '$w_3$'], fontsize=10)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_yticks([1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
        ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=8)
        ax.tick_params(axis='x', which='both', bottom=True)
        ax.grid(False)

        # Draw outer gray box
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        rect = patches.Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            linewidth=0.4,
            edgecolor='gray',
            facecolor='none',
            zorder=10,
            transform=ax.transData,
            clip_on=False
        )
        ax.add_patch(rect)

        # Prepare legend
        handles, labels = ax.get_legend_handles_labels()
        selected_handles = []
        selected_labels = []

        for val in legend_vals:
            closest_idx = np.argmin([abs(float(l) - val) for l in labels])
            selected_handles.append(handles[closest_idx])
            if class_column == 'Objective_2':
                formatted = f"{int(round(float(labels[closest_idx]) / 1000))}k"
            elif abs(float(labels[closest_idx])) < 100:
                formatted = f"{float(labels[closest_idx]):.2f}"
            else:
                formatted = f"{int(round(float(labels[closest_idx])))}"
            selected_labels.append(formatted)

        ax.legend(
            selected_handles, selected_labels,
            fontsize=6.5, loc='upper left', bbox_to_anchor=(1.05, 1),
            frameon=True, title=None
        )

    plt.tight_layout()
    fig.savefig('../results/plots/fig8_combined_PCP.png', dpi=500, bbox_inches="tight")
    plt.show()
    # plt.close()

