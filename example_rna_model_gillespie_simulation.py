import gillespy2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pysindy as ps
from pysindy.optimizers import STLSQ
import cvxpy

# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


# The code is basically a copy from: https://pypi.org/project/gillespy2/
def create_gillespie_rna_model(M, c, max_time):
    """Gillespie algorithm for simulating one realization of RNA splicing dynamics.

    :param M: Initial state, numpy array with shape (2, 1).
    :param c: Vector of stochastic rate constants, list with length 3.
    :param max_time: Length of simulation time span, one number.

    :return:
        T - One-dimensional numpy array containing the reaction occurrence times.
        X - Two dimensional numpy array, where the rows contain the system state after each reaction.
    """
    # First call the gillespy2.Model initializer.
    model = gillespy2.Model(name='mRNA-splicing')

    # Define parameters for the rates of the reactions.
    c1 = gillespy2.Parameter(name='c1_transcription', expression=c[0])
    c2 = gillespy2.Parameter(name='c2_degrading', expression=c[1])
    c3 = gillespy2.Parameter(name='c3_expression', expression=c[2])
    model.add_parameter([c1, c2, c3])

    # Define variables for the molecular species representing M and D.
    x1 = gillespy2.Species(name='X1_unspliced_mRNA', initial_value=int(M[0]))
    x2 = gillespy2.Species(name='X2_spliced_mRNA', initial_value=int(M[1]))
    model.add_species([x1, x2])

    # The list of reactants and products for a Reaction object are each a
    # Python dictionary in which the dictionary keys are Species objects
    # and the values are stoichiometries of the species in the reaction.
    r1 = gillespy2.Reaction(name="r1_transcription", rate=c1, reactants={}, products={x1: 1})
    r2 = gillespy2.Reaction(name="r2_degrading", rate=c2, reactants={x2: 1}, products={})
    r3 = gillespy2.Reaction(name="r3_expression", rate=c3, reactants={x1: 1}, products={x2: 1})
    model.add_reaction([r1, r2, r3])

    # Set the timespan for the simulation.
    tspan = gillespy2.TimeSpan.linspace(t=max_time, num_points=1000)
    model.timespan(tspan)
    return model


def interpolate_to_fixed_timesteps(res, new_time_step):
    # Extract t, x, and y values
    t = res[:, 0]
    x = res[:, 1]
    y = res[:, 2]

    # Create a new time array with fixed time steps
    new_t = np.arange(min(t), max(t), new_time_step)

    # Interpolate x and y values for the new time steps
    x_interp = interp1d(t, x, kind='linear', fill_value='extrapolate')
    y_interp = interp1d(t, y, kind='linear', fill_value='extrapolate')

    # Generate new x and y values
    new_x = x_interp(new_t)
    new_y = y_interp(new_t)

    # Combine the new x, and new y into one array
    new_res = np.vstack((new_x, new_y)).T

    return new_res


def deterministic_rna(t, y, c1, c2, c3):
    y1, y2 = y
    return [c1 - c3 * y1, -c2 * y2 + c3 * y1]


def plot_model_prediction(model, x_test, dt, max_time):
    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test)

    # Compute derivatives with a finite difference method, for comparison
    x_dot_test_computed = model.differentiate(x_test, t=dt)

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(np.arange(0, max_time, dt), x_dot_test_computed[:, i], "k", label="numerical derivative")
        axs[i].plot(np.arange(0, max_time, dt), x_dot_test_predicted[:, i], "r--", label="model prediction")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
    fig.show()


def get_true_rna_model(M, c, max_time, dt):
    t_eval = np.linspace(0, max_time, 200)
    sol = solve_ivp(deterministic_rna, [0, max_time], [M[0][0], M[1][0]], t_eval=t_eval, args=c)
    T = sol.t
    X = sol.y.T

    return interpolate_to_fixed_timesteps(np.vstack((T, X[:, 0], X[:, 1])).T, dt)


def model_with_equality_constrains(res_interpolated, dt, constraint_rhs, constraint_lhs):
    # Define the optimizer
    optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs)
    model_equality_constrained = ps.SINDy(optimizer=optimizer, feature_library=library)

    for x_data in res_interpolated:
        model_equality_constrained.fit(x_data, t=dt)
    print("Equality Constrained Model")
    model_equality_constrained.print()


if __name__ == "__main__":
    dt = 0.1

    M = np.array([[100, 0]]).T  # has shape (2, 1)
    c = [10, 0.25, 0.5]
    max_time = 30

    print("Optimal Model")
    print(f"(x0)' = {c[0]} 1 - {c[2]} x0")
    print(f"(x1)' = {c[2]} x0 - {c[1]} x1")

    # Create the simulation_data
    results = np.array(create_gillespie_rna_model(M, c, max_time).run(number_of_trajectories=10).to_array())

    # Interpolate the data to be in fixed time steps
    res_interpolated = [interpolate_to_fixed_timesteps(res, dt) for res in results]

    # Define the model with the custom optimizer
    model_unconstrained = ps.SINDy()

    for x_data in res_interpolated:
        model_unconstrained.fit(x_data, t=dt)
    print("Unconstrained Model:")
    model_unconstrained.print()

    # True underlying model for reference
    x_true = get_true_rna_model(M, c, max_time, dt)

    print("Model score against true model: %f" % model_unconstrained.score(x_true, t=dt))
    plot_model_prediction(model_unconstrained, x_true, dt, max_time)

    # Create Model again with constraints
    library = ps.PolynomialLibrary()
    library.fit([ps.AxesArray(x_data, {"ax_sample": 0, "ax_coord": 1})])
    n_features = library.n_output_features_
    print(f"Features ({n_features}):", library.get_feature_names())

    # Set
    n_targets = x_true.shape[1]
    constraint_rhs = np.array([0, 28])

    # One row per constraint, one column per coefficient
    constraint_lhs = np.zeros((2, n_targets * n_features))

    # 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
    constraint_lhs[0, 1] = 1
    constraint_lhs[0, 2] = 1

    # 1 * (x0 coefficient) = 28
    constraint_lhs[1, 1 + n_features] = 1

    # Define the optimizer
    optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs)
    model_equality_constrained = ps.SINDy(optimizer=optimizer, feature_library=library)

    for x_data in res_interpolated:
        model_equality_constrained.fit(x_data, t=dt)
    print("Equality Constrained Model")
    model_equality_constrained.print()

    print("Model score against true model: %f" % model_equality_constrained.score(x_true, t=dt))
    plot_model_prediction(model_equality_constrained, x_true, dt, max_time)
