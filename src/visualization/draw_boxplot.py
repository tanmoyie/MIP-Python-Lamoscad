import seaborn as sns
import matplotlib.pyplot as plt


def draw_boxplot(df_boxplot):
    # Prepare stacked DataFrames
    coverage_df_stacked = df_boxplot[['Coverage%_c', 'Coverage%_p']].copy()
    coverage_df_stacked.columns = ['Current', 'Proposed']
    coverage_df_stacked = coverage_df_stacked.stack().reset_index()
    coverage_df_stacked.columns = ['Index', 'Facility', 'Coverage']

    cost_df_stacked = df_boxplot[['Obj2_c', 'Obj2_p']].copy()
    cost_df_stacked.columns = ['Current', 'Proposed']
    cost_df_stacked = cost_df_stacked.stack().reset_index()
    cost_df_stacked.columns = ['Index', 'Facility', 'Cost']

    time_df_stacked = df_boxplot[['Mean_RT_c', 'Mean_RT_p']].copy()
    time_df_stacked.columns = ['Current', 'Proposed']
    time_df_stacked = time_df_stacked.stack().reset_index()
    time_df_stacked.columns = ['Index', 'Facility', 'Response Time']

    # Set theme and plotting parameters
    sns.set_theme(style="white")
    plt.rcParams['axes.edgecolor'] = 'lightgray'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4))
    alpha_val = 0.7

    # Box and strip plots
    sns.boxplot(data=coverage_df_stacked, ax=ax1, x='Facility', y='Coverage',
                boxprops=dict(alpha=alpha_val), palette=["g", "yellow"])
    sns.stripplot(data=coverage_df_stacked, ax=ax1, x='Facility', y='Coverage',
                  alpha=alpha_val)

    sns.boxplot(data=cost_df_stacked, ax=ax2, x='Facility', y='Cost',
                boxprops=dict(alpha=alpha_val), palette=["g", "yellow"])
    sns.stripplot(data=cost_df_stacked, ax=ax2, x='Facility', y='Cost',
                  alpha=alpha_val)

    sns.boxplot(data=time_df_stacked, ax=ax3, x='Facility', y='Response Time',
                boxprops=dict(alpha=alpha_val), palette=["g", "yellow"])
    sns.stripplot(data=time_df_stacked, ax=ax3, x='Facility', y='Response Time',
                  alpha=alpha_val)

    # Label formatting
    ax1.set_ylabel('Spill Coverage (%)')
    ax2.set_ylabel('Cost Objective Value')
    ax3.set_ylabel('Mean Response Time (in hr)')
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels(['Current', 'Proposed'])
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=30)
    plt.tight_layout()
    fig.savefig('../results/plots/Fig10 boxplot.png', transparent=False, dpi=500)


