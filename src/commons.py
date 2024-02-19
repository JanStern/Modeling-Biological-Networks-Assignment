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
