# Modelin Biological Networks Assignment
#
# Jan Sternagel, 102177941
import json
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

from src.commons.data_transformation import read_data
from src.visuals.plots import show_gene_expression_over_time_in_one_plot


def main():
    # Load the data
    t, data_gen_exp_t = read_data('data/data_original.json')

    # Show how the gene expression over time varies
    # show_gene_expression_over_time_in_one_plot(t, data_gen_exp_t)

    t_data = np.array(t)
    Y_data = np.array(list(data_gen_exp_t.values()))

    print(Y_data)


if __name__ == '__main__':
    main()


def odefun(x, k):
    """Differential function of the ODE system.

    :param x: a list with length 2, i.e. values of the dynamical variables
    :param k: parameters, a list with length 2
    :returns: dx/dt, a list with length 2
    """
    dx1_dt = 2.0 * x[0] - k[0] * x[0] * x[1]
    dx2_dt = k[0] * x[0] * x[1] - k[1] * x[1]
    dx_dt = [dx1_dt, dx2_dt]
    return dx_dt


def target_function(k, Y_data, t_data):
    """Sum of squared errors.
    :param k: Rate parameters. A list with length 2.
    :return: A real number.
    """
    x0 = Y_data.T[0]
    fun = lambda t, y: odefun(x=y, k=k)  # differential function with fixed k values
    out = solve_ivp(fun, y0=x0, t_span=[min(t_data), max(t_data)], t_eval=t_data)  # solve the initial value problem
    x_data = out.y.T

    # Compute the sum of squared errors
    error = np.linalg.norm(Y_data - x_data, axis=1)  # Compute Euclidean norm row-wise
    return np.sum(error ** 2)
