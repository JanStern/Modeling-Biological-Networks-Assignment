import gillespy2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pysindy as ps
from pysindy.optimizers import STLSQ

# Seed the random number generators for reproducibility
np.random.seed(100)


def create_gillespie_lotka_voltera(M, c, max_time):
    """Gillespie algorithm for simulating one realization of RNA splicing dynamics.

    :param M: Initial state, numpy array with shape (2, 1).
    :param c: Vector of stochastic rate constants, list with length 3.
    :param max_time: Length of simulation time span, one number.

    :return:
        T - One-dimensional numpy array containing the reaction occurrence times.
        X - Two dimensional numpy array, where the rows contain the system state after each reaction.
    """
    # First call the gillespy2.Model initializer.
    model = gillespy2.Model(name='lotka-voltera')

    # Define parameters for the rates of the reactions.
    c1 = gillespy2.Parameter(name='c1_transcription', expression=c[0])
    c2 = gillespy2.Parameter(name='c2_degrading', expression=c[1])
    c3 = gillespy2.Parameter(name='c3_expression', expression=c[2])
    model.add_parameter([c1, c2, c3])

    # Define variables for the molecular species representing M and D.
    x0 = gillespy2.Species(name='x0', initial_value=int(M[0]))
    x1 = gillespy2.Species(name='X1', initial_value=int(M[1]))
    model.add_species([x0, x1])

    # The list of reactants and products for a Reaction object are each a
    # Python dictionary in which the dictionary keys are Species objects
    # and the values are stoichiometries of the species in the reaction.
    r1 = gillespy2.Reaction(name="x0_reproduction", rate=c1, reactants={x0: 1}, products={x0: 2})
    r2 = gillespy2.Reaction(name="x1_eats_x0", rate=c2, reactants={x0: 1, x1: 1}, products={x1: 2})
    r3 = gillespy2.Reaction(name="x1_dies", rate=c3, reactants={x1: 1}, products={})
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


def lotka_voltera(t, y, c1, c2, c3):
    y1, y2 = y
    return [c1 * y1 - c2 * y1 * y2, c2 * y1 * y2 - c3 * y2]


def get_true_lotka_voltera(M, c, max_time):
    t_eval = np.linspace(0, max_time, 200)
    sol = solve_ivp(lotka_voltera, [0, max_time], [M[0][0], M[1][0]], t_eval=t_eval, args=c)
    T = sol.t
    X = sol.y.T

    return interpolate_to_fixed_timesteps(np.vstack((T, X[:, 0], X[:, 1])).T, dt)


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


def plot_value_space(x_true, x_test):
    plt.plot(x_true[:, 0], x_true[:, 1], "k", label="true_distribution")
    plt.show()


if __name__ == "__main__":
    dt = 0.1

    M = np.array([[80, 150]]).T  # has shape (2, 1)
    c = [1.0, 0.01, 0.7]
    max_time = 50

    # Create the simulation_data
    results = np.array(create_gillespie_lotka_voltera(M, c, max_time).run(number_of_trajectories=10).to_array())

    # Interpolate the data to be in fixed time steps
    res_interpolated = [interpolate_to_fixed_timesteps(res, dt) for res in results]

    # Define the model with the custom optimizer
    model = ps.SINDy()

    for x_data in res_interpolated:
        model.fit(x_data, t=dt)
    model.print()

    x_true = get_true_lotka_voltera(M, c, max_time)

    print("Model score against true model: %f" % model.score(x_true, t=dt))

    plot_value_space(x_true, None)
