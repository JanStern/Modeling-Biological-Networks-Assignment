import json
from typing import Any, Tuple, Dict


def separate_time_axis_from_data(data: dict[str, list]):
    """
    Remove the time axis from the data.
    The time axis needs to be named 't'!
    """
    t: list = data.pop("t")
    return t, data


def read_data(file_path: str) -> tuple[list, dict[str, list]]:
    with open(file_path, 'r') as f:
        data_gen_exp_t = json.load(f)

    return separate_time_axis_from_data(data_gen_exp_t)


def calculate_confusion_matrix_from_model(model_predicted: dict[str, list[float]],
                                          compare_aginst_switch_off_sereies: bool = True,
                                          consider_inhibition: bool = True,
                                          undirected_model: bool = False):
    """

    """
    model_switch_off_series = {
        'SWI5': [None, 0, -1, 0, -1],
        'CBF1': [0, None, 0, 0, 0],
        'GAL4': [1, 1, None, 1, 0],
        'GAL80': [0, 0, 0, None, 0],
        'ASH1': [0, 0, 0, 0, None]
    }

    model_true_model = {
        'SWI5': [None, 1, 0, 1, 1],
        'CBF1': [0, None, 1, 0, 0],
        'GAL4': [1, 0, None, -0.5, 0],
        'GAL80': [0, 0, -0.5, None, 0],
        'ASH1': [0, -1, 0, 0, None]
    }

    undirected_model_switch_off_series = {
        'SWI5': [None, 0, 1, 0, 1],
        'CBF1': [0, None, 1, 0, 0],
        'GAL4': [1, 1, None, 1, 0],
        'GAL80': [0, 0, 1, None, 0],
        'ASH1': [1, 0, 0, 0, None]
    }

    undirected_model_true_model = {
        'SWI5': [None, 1, 1, 1, 1],
        'CBF1': [1, None, 1, 0, 1],
        'GAL4': [1, 1, None, 1, 0],
        'GAL80': [1, 0, 1, None, 0],
        'ASH1': [1, 1, 0, 0, None]
    }

    if undirected_model:
        model_compare = undirected_model_switch_off_series if compare_aginst_switch_off_sereies else undirected_model_true_model
        # Also undirected models can't differentiate between inhibition or excitement
        consider_inhibition = False
    else:
        model_compare = model_switch_off_series if compare_aginst_switch_off_sereies else model_true_model

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for variable in model_compare:
        connections_compare = model_compare[variable]
        connections_predicted = model_predicted[variable]

        for comp, pred in zip(connections_compare, connections_predicted):

            if comp is None:
                # We don't know what the model is doing when
                continue

            elif comp == 0:
                if pred == 0:
                    true_negatives += 1
                else:
                    false_positives += 1

            elif comp < 0:
                if pred < 0:
                    true_positives += 1
                elif pred == 0:
                    false_negatives += 1
                elif pred > 0 and not consider_inhibition:
                    true_positives += 1

            elif comp > 0:
                if pred > 0:
                    true_positives += 1
                elif pred == 0:
                    false_negatives += 1
                elif pred < 0 and not consider_inhibition:
                    true_positives += 1

    try:
        fpr: float = false_positives / (false_positives + true_negatives)
    except ZeroDivisionError:
        fpr = 1.0

    try:
        tpr: float = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        tpr = 0.0
    return fpr, tpr


def apply_threshold(matrix_original, threshold):
    matrix = matrix_original.copy()
    elements = matrix.columns
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements):
            if i == j:
                continue

            val = matrix.iloc[i, j]
            if abs(val) < threshold:  # Check if the correlation meets the threshold
                matrix.iloc[i, j] = 0.0
    return matrix
