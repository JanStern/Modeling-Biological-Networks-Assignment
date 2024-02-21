import re
import numpy as np
import pysindy as ps


def pruned_model_optimizer_equality(prior_knowledge):
    """
    n_targets = number of variables that the system has
    prior_knowledge = define the ODE elements that you are certain about
        e.g. prior_knowledge = {
                                    # "feature_names":  ['1',   'x0',   'x1',   'x0^2', 'x0 x1',    'x1^2'],
                                    "x0":               [None,  None,   0,      0,      0,          0],
                                    "x1":               [0,     None,   None,   0,      0,          0]
                                }
        None: unknown connection -> no constrains for the model
        Number: This constant of the ODE needs to be this value

    """
    n_targets = len(prior_knowledge.keys())  # Every key is a target
    n_features = len(list(prior_knowledge.values())[0])  # All features that could exist and could have a constant

    # Set up the constraints for the optimizer
    constraint_rhs = np.array([])
    constraint_lhs = []
    for i_target, measured_variable in enumerate(prior_knowledge):

        constants_knowledge = prior_knowledge[measured_variable]
        for i_feature, const_val in enumerate(constants_knowledge):

            # None -> No knowledge about the connection. Don't add any constraints
            if const_val is None or isinstance(const_val, str):
                continue

            # If we know something we need to append constraint_rhs with the value this constant should have
            constraint_rhs = np.append(constraint_rhs, np.array([const_val]))

            # The constraint_lhs needs to be the position of the model
            constraint = np.zeros((n_targets, n_features))
            constraint[i_target, i_feature] = 1
            constraint = np.array(constraint.flatten())

            constraint_lhs.append(constraint)

    optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, constraint_lhs=np.array(constraint_lhs))
    return optimizer


def pruned_model_optimizer_inequality(prior_knowledge, eps: float = 1e-6):
    """
    n_targets = number of variables that the system has
    prior_knowledge = define the ODE elements that you are certain about
        e.g. prior_knowledge = {
                                    # "feature_names":  ['1',   'x0',   'x1',   'x0^2', 'x0 x1',    'x1^2'],
                                    "x0":               [None,  None,   0,      0,      0,          0],
                                    "x1":               [0,     None,   None,   0,      0,          0]
                                }
        None: unknown connection -> no constrains for the model
        Number: This constant of the ODE needs to be this value

    """
    n_targets = len(prior_knowledge.keys())  # Every key is a target
    n_features = len(list(prior_knowledge.values())[0])  # All features that could exist and could have a constant

    # Set up the constraints for the optimizer
    constraint_rhs = np.array([])
    constraint_lhs = []
    for i_target, measured_variable in enumerate(prior_knowledge):

        constants_knowledge = prior_knowledge[measured_variable]
        for i_feature, const_val in enumerate(constants_knowledge):

            # None -> No knowledge about the connection. Don't add any constraints
            if const_val is None:
                continue

            if const_val == 0:
                # Also add restriction from the bottom
                constraint_rhs = np.append(constraint_rhs, np.array([eps]))
                constraint = np.zeros((n_targets, n_features))
                constraint[i_target, i_feature] = -1
                constraint_lhs.append(np.array(constraint.flatten()))

            # coefficient <= const_val + eps
            constraint_rhs = np.append(constraint_rhs, np.array([const_val + eps]))  # Also add a small offset

            # The constraint_lhs needs to be the position of the model
            constraint = np.zeros((n_targets, n_features))
            constraint[i_target, i_feature] = 1

            constraint_lhs.append(np.array(constraint.flatten()))

    optimizer = ps.ConstrainedSR3(
            constraint_rhs=constraint_rhs,
            constraint_lhs=np.array(constraint_lhs),
            inequality_constraints=True,
            thresholder="l1",
            tol=1e-7,
            threshold=10,
            max_iter=10000
    )
    return optimizer


def translate_string(input_string, translation_dict):
    # Use a regular expression to replace each key with its corresponding value
    for key, value in translation_dict.items():
        input_string = re.sub(r'\b' + key + r'\b', value, input_string)
    return input_string


def translate_model_to_spare_array(rna_to_element_translation, feature_names, model_description):
    translated_model = {f"x{i}": [] for i in range(len(model_description))}
    element_to_rna_translation = {value: key for key, value in rna_to_element_translation.items()}

    for rna_name, target_values in model_description.items():
        target = rna_to_element_translation[rna_name]

        target_features = []
        for feature_name in feature_names:
            translated_feature_name = translate_string(feature_name, element_to_rna_translation)

            val = 0

            if translated_feature_name in target_values:
                val = None

            # Check if the order needs to be switched
            if " " in translated_feature_name and " ".join(
                    list(reversed(translated_feature_name.split(" ")))) in target_values:
                val = None

            target_features.append(val)

        translated_model[target] = target_features

    return translated_model


def find_indexes_with_substring(arr, substring):
    """
    Find indexes of elements containing the specified substring in an array.

    Parameters:
    - arr: List[str]. The array to search through.
    - substring: str. The substring to look for in the array's elements.

    Returns:
    - List[int]. A list of indexes of the elements that contain the substring.
    """
    indexes = [i for i, element in enumerate(arr) if substring in element]
    return indexes

def process_coefficient_matrix(coefficient_matrix, feature_biological_names, library_features):
    coefficient_matrix_processed = {}
    for i, element in enumerate(feature_biological_names):
        values = []
        for j, _ in enumerate(feature_biological_names):
            indices = find_indexes_with_substring(library_features, f"x{j}")
            values.append(sum(coefficient_matrix[i, indices]))

        coefficient_matrix_processed[element] = values

    return coefficient_matrix_processed
