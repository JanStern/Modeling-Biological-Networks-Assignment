# All matplotlib plots are stored here
import matplotlib.pyplot as plt

GENE_COLORS = {"SWI5": "blue", "CBF1": "green", "GAL4": "red", "GAL80": "turquoise", "ASH1": "magenta"}


def show_gene_expression_over_time_in_one_plot(t: list, data_gen_exp_t: dict[str, list], **kwargs) -> None:
    """
    The gene expression of all Genes is shown in one figure

    The data format always needs to follow the example given in data_original.json:
        data = {
                    gen_1: [0.12, 0.28, 0.23, 0.84, 0.55],
                    gen_2: [0.12, 0.28, 0.23, 0.84, 0]
                }
    """
    # Creating the plot
    plt.figure(figsize=(10, 6))  # Setting the figure size

    for gen_name, gen_expression_values in data_gen_exp_t.items():
        plt.plot(t, gen_expression_values, label=gen_name, color=GENE_COLORS[gen_name])

    # Adding labels and title
    plt.xlabel('Time (min)', fontsize=14, fontweight='bold')  # X-axis label
    plt.ylabel('Gene Expression (arbitrary units)', fontsize=14, fontweight='bold')  # Y-axis label
    plt.title('Time-course Gene Expression', fontsize=16, fontweight='bold')  # Plot title

    # Customizing the tick marks for readability
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Adding a legend
    plt.legend(loc='best', fontsize=12)

    plt.xlim(min(t), max(t))

    # Display the plot
    plt.show()
