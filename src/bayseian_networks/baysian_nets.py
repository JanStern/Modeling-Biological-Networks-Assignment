import pymc3 as pm
from src.commons.data_transformation import read_data

t, gene_expression_data = read_data('data/data_original.json')

# Define the number of genes and time points based on your data shape
num_genes, num_time_points = gene_expression_data.shape

with pm.Model() as gene_network_model:
    # Define priors for initial expression levels of each gene
    initial_expression = pm.Normal('initial_expression', mu=0, sd=1, shape=num_genes)

    # Define priors for the rate of change of expression over time for each gene
    expression_change_rate = pm.Normal('expression_change_rate', mu=0, sd=1, shape=(num_genes, num_genes))

    # Build the time-series model for gene expression levels
    expression_level = initial_expression
    for t in range(1, num_time_points):
        expression_level = expression_level + pm.math.dot(expression_change_rate, gene_expression_data[:, t - 1])
        # Add Gaussian noise to model the observation at each time point
        gene_expression_data[:, t] = pm.Normal(f'observation_{t}', mu=expression_level, sd=0.1,
                                               observed=gene_expression_data[:, t])

    # Perform MCMC
    trace = pm.sample(10, tune=500, return_inferencedata=True)

# After sampling, you can inspect the trace to analyze the posterior distributions
# for the initial_expression and expression_change_rate parameters.


# # Assume gene_expression_data is your observed data matrix
# # initial_expression_levels is the initial expression levels for each gene
# initial_expression_levels = gene_expression_data[:, 0]
#
# with pm.Model() as prediction_model:
#     # Priors for the rate of change for each gene
#     rates_of_change = pm.Normal('rates_of_change', mu=0, sd=1, shape=num_genes)
#
#     # Linear prediction of gene expression over time for each gene
#     for i in range(num_genes):
#         expression_over_time = initial_expression_levels[i] + rates_of_change[i] * num_time_points
#
#         # Likelihood for each gene's expression over time
#         pm.Normal(f'gene_{i}_expression', mu=expression_over_time, sd=0.1, observed=gene_expression_data[i])
#
#     # Sample from the posterior to fit the model
#     trace = pm.sample(1000, tune=500)
#
#     # Use the trace to generate predictions
#     # Here we're taking the mean of the posterior for rate of change for simplicity
#     mean_rates_of_change = np.mean(trace['rates_of_change'], axis=0)
#     predicted_expression = initial_expression_levels + mean_rates_of_change * t[1:]
