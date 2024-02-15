# Modelin Biological Networks Assignment
#
# Jan Sternagel, 102177941

import numpy as np
import pysindy as ps
from src.commons.data_transformation import read_data
from src.gradient_estimation.pysindy_gradiet_estimation import pruned_model_optimizer_inequality, \
    pruned_model_optimizer_equality, translate_model_to_spare_array
from src.visuals.plots import show_gene_expression_over_time_in_one_plot, plot_model_prediction, \
    plot_scores_over_threshold, format_model_for_graph_plot, create_graph_from_model, draw_graph

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


def main():
    # Load the data
    t, data_gen_exp_t = read_data('data/data_original.json')

    # Show how the gene expression over time varies
    # show_gene_expression_over_time_in_one_plot(t, data_gen_exp_t)

    dt = 10  # min
    max_time = max(t)

    t_data = np.array(t)
    x_data = np.array(list(data_gen_exp_t.values())).T

    # Define the library and identify all the features
    library = ps.PolynomialLibrary()
    library.fit([ps.AxesArray(x_data, {"ax_sample": 0, "ax_coord": 1})])
    n_features = library.n_output_features_
    print(f"Features ({n_features}):", library.get_feature_names())

    # 1. Start of with an unconstrained model to establish a baseline
    model_unconstrained = ps.SINDy()
    model_unconstrained.fit(x_data, t=dt)
    print("Unconstrained Model:")
    model_unconstrained.print()

    print("Model score against true model: %f" % model_unconstrained.score(x_data, t=dt))
    plot_model_prediction(model_unconstrained, x_data, dt, max_time)

    # 2. Experiment with the threshold
    threshold_scan = np.linspace(0, 1, 21)
    scores = []
    for i, threshold in enumerate(threshold_scan):
        optimizer = ps.STLSQ(threshold=threshold)
        model_threshold = ps.SINDy(optimizer=optimizer, feature_library=library)
        model_threshold.fit(x_data, t=dt)
        if i == 0:
            model_threshold.print()
        scores.append(model_threshold.score(x_data, t=dt))
    plot_scores_over_threshold(threshold_scan, scores)

    # 3. Add prior biological knowledge about the system
    rna_to_element_translation: dict = {
        "SWI5": "x0",
        "CBF1": "x1",
        "GAL4": "x2",
        "GAL80": "x3",
        "ASH1": "x4"
    }

    model_linear_no_further_constants_no_combinations = {
        "SWI5": ["SWI5", "GAL4"],
        "CBF1": ["SWI5", "ASH1"],
        "GAL4": ["CBF1", "GAL80"],
        "GAL80": ["1", "SWI5"],
        "ASH1": ["SWI5"]
    }

    model_linear_constants_no_combinations = {
        "SWI5": ["1", "SWI5", "GAL4"],
        "CBF1": ["1", "SWI5", "ASH1"],
        "GAL4": ["1", "CBF1", "GAL80"],
        "GAL80": ["1", "SWI5"],
        "ASH1": ["1", "SWI5"]
    }

    model_linear_no_further_constants_combinations = {
        "SWI5": ["SWI5", "GAL4"],
        "CBF1": ["SWI5", "ASH1", "SWI5 ASH1"],
        "GAL4": ["CBF1", "GAL80"],
        "GAL80": ["1", "SWI5"],
        "ASH1": ["SWI5"]
    }

    model_linear_constants_combinations = {
        "SWI5": ["1", "SWI5", "GAL4"],
        "CBF1": ["1", "SWI5", "ASH1", "SWI5 ASH1"],
        "GAL4": ["1", "CBF1", "GAL80"],
        "GAL80": ["1", "SWI5"],
        "ASH1": ["1", "SWI5"]
    }

    # Quadratic
    model_quadratic_no_further_constants_no_combinations = {
        "SWI5": ["SWI5", "GAL4", "SWI5^2", "GAL4^2"],
        "CBF1": ["SWI5", "ASH1", "SWI5^2", "ASH1^2"],
        "GAL4": ["CBF1", "GAL80", "CBF1^2", "GAL80^2"],
        "GAL80": ["1", "SWI5", "SWI5^2"],
        "ASH1": ["SWI5", "SWI5^2"]
    }

    model_quadratic_constants_no_combinations = {
        "SWI5": ["1", "SWI5", "GAL4", "SWI5^2", "GAL4^2"],
        "CBF1": ["1", "SWI5", "ASH1", "SWI5^2", "ASH1^2"],
        "GAL4": ["1", "CBF1", "GAL80", "CBF1^2", "GAL80^2"],
        "GAL80": ["1", "SWI5", "SWI5^2"],
        "ASH1": ["1", "SWI5", "SWI5^2"]
    }

    model_quadratic_no_further_constants_combinations = {
        "SWI5": ["SWI5", "GAL4", "SWI5^2", "GAL4^2"],
        "CBF1": ["SWI5", "ASH1", "SWI5 ASH1", "SWI5^2", "ASH1^2"],
        "GAL4": ["CBF1", "GAL80", "CBF1^2", "GAL80^2"],
        "GAL80": ["1", "SWI5", "SWI5^2"],
        "ASH1": ["SWI5", "SWI5^2"]
    }

    model_quadratic_constants_combinations = {
        "SWI5": ["1", "SWI5", "GAL4", "SWI5^2", "GAL4^2"],
        "CBF1": ["1", "SWI5", "ASH1", "SWI5 ASH1", "SWI5^2", "ASH1^2"],
        "GAL4": ["1", "CBF1", "GAL80", "CBF1^2", "GAL80^2"],
        "GAL80": ["1", "SWI5", "SWI5^2"],
        "ASH1": ["1", "SWI5", "SWI5^2"]
    }

    translated_model = translate_model_to_spare_array(rna_to_element_translation, library.get_feature_names(),
                                                      model_linear_constants_combinations)

    model_equality_constrained = ps.SINDy(
            optimizer=pruned_model_optimizer_equality(translated_model),
            feature_library=library
    )
    model_equality_constrained.fit(x_data, t=dt)
    print("Equality Constrained Model")
    model_equality_constrained.print()
    print("Model score against true model: %f" % model_equality_constrained.score(x_data, t=dt))
    plot_model_prediction(model_equality_constrained, x_data, dt, max_time)

    # Create the graph from model equations
    G = create_graph_from_model(format_model_for_graph_plot(model_equality_constrained, library))

    # Draw the graph with adapted edge thickness and color
    # draw_graph(G)


if __name__ == '__main__':
    main()
