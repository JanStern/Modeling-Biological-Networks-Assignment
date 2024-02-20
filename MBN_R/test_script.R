# Load the jsonlite package for reading JSON files
library(jsonlite)
library(combinat)
library(igraph)
library(lavaan)
# Load the data from the JSON file
json_data <- fromJSON("data_original.json", )

# Convert the loaded JSON data into a data frame
data_frame <- as.data.frame(json_data)

# Define the SEM model
sem_model <- '
  # Regression equations
  GAL4 ~ GAL80
  GAL80 ~ SWI5
'

# Fit the model
fit <- sem(sem_model, data = data_frame)

# Summary of the model fit
model_summary <- summary(fit)

print(model_summary)



# Load deSolve package
library(deSolve)

# Define the differential equations
# In this example, y is a vector where y[1] is the prey population and y[2] is the predator population
# params is a vector of parameters: rate of prey growth, rate of predation, rate of predator growth, rate of predator death
lotka_volterra <- function(t, y, params) {
  with(as.list(c(y, params)), {
    dprey <- prey_growth * y[1] - predation_rate * y[1] * y[2]
    dpredator <- predator_growth * y[1] * y[2] - predator_death * y[2]
    list(c(dprey, dpredator))
  })
}

# Initial conditions
initial_conditions <- c(y1 = 10, y2 = 5)  # For example, 10 prey and 5 predators

# Parameters
params <- c(prey_growth = 1.1, predation_rate = 0.4, predator_growth = 0.1, predator_death = 0.4)

# Time
times <- seq(0, 50, by = 0.1)  # From time 0 to 50 in steps of 0.1

# Solve the differential equations
output <- ode(y = initial_conditions, times = times, func = lotka_volterra, parms = params)

# Plot the results
plot(output, main = "Predator-Prey Dynamics")