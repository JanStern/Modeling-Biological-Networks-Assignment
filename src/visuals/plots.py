# All matplotlib plots are stored here
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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


def plot_model_prediction(model, x_test, dt, max_time):
    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test)

    # Compute derivatives with a finite difference method, for comparison
    x_dot_test_computed = model.differentiate(x_test, t=dt)

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(np.arange(0, max_time + 1, dt), x_dot_test_computed[:, i], "k", label="numerical derivative")
        axs[i].plot(np.arange(0, max_time + 1, dt), x_dot_test_predicted[:, i], "r--", label="model prediction")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
    fig.show()


def plot_scores_over_threshold(threshold_scan, scores):
    plt.plot(threshold_scan, scores, label="Score")
    plt.legend()
    plt.show()


def plot_coefs_over_threshold(threshold_scan, coefs):
    coefs = np.array(coefs)

    fig, ax = plt.subplots(coefs.shape[1], coefs.shape[2], figsize=(30, 20))
    for i_target in range(coefs.shape[1]):
        for i_feature in range(coefs.shape[2]):
            ax[i_target, i_feature].plot(threshold_scan, coefs[:, i_target, i_feature])
    plt.legend()
    plt.xlabel("Threshold value")
    plt.ylabel("Score of model")
    plt.show()


def create_graph_from_model(model_equations):
    G = nx.DiGraph()

    # Add nodes and edges based on model equations
    for target, interactions in model_equations.items():
        for source, rate in interactions:
            if source == target:
                continue
            if rate == 0.0:
                continue

            if source == "1":
                source = f"{target} Source"

            G.add_edge(source, target, rate=rate)

    return G


def draw_graph(G):
    # Use a layout for dynamic structuring
    pos = nx.spring_layout(G, seed=42)  # For consistent layout across runs

    # Draw the graph with customized edges
    for edge in G.edges(data=True):
        source, target, data = edge
        rate = data['rate']

        # Determine color based on rate sign
        color = 'green' if rate > 0 else 'red'

        # Adjust thickness based on the absolute value of rate, ensuring a minimum thickness
        thickness = max(abs(rate) * 2, 0.5)

        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], width=thickness, edge_color=color, arrowsize=50)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'rate')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Dynamically Structured Graph from PySINDy Model")
    plt.axis('off')
    plt.show()


def format_model_for_graph_plot(pysindy_model, library):
    coef = np.array(pysindy_model.coefficients())

    feature_names = library.get_feature_names()

    model_equations = {}
    for i in range(coef.shape[0]):
        target = f"x{i}"
        model_equations[target] = []
        for j in range(coef.shape[1]):
            model_equations[target].append((feature_names[j], round(coef[i, j], 3)))

    return model_equations
