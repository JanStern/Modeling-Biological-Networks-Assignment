# Load necessary libraries
library(jsonlite)
library(igraph)
library(infotheo)

# Load and preprocess data
data_path <- "data_original.json"
data <- fromJSON(data_path)

# Convert list to data frame for easier manipulation
data_df <- as.data.frame(data)

# Discretize the data (required for mutual information calculation)
discretized_data <- data_df
for(i in 1:ncol(discretized_data)) {
  discretized_data[[i]] <- discretize(discretized_data[[i]])
}

# Calculate mutual information
calc_mutual_information <- function(data) {
  num_vars <- ncol(data)
  mi_matrix <- matrix(0, nrow = num_vars, ncol = num_vars)
  for (i in 1:num_vars) {
    for (j in i:num_vars) {
      mi <- mutinformation(data[, i], data[, j])
      mi_matrix[i, j] <- mi
      mi_matrix[j, i] <- mi
    }
  }
  colnames(mi_matrix) <- names(data)
  return(mi_matrix)
}

mi_matrix <- calc_mutual_information(discretized_data)

# Filter connections based on a threshold and create a network
threshold <- 0.0201355136 # Set your mutual information threshold here
graph <- graph.adjacency(mi_matrix > threshold, weighted = TRUE)
graph <- simplify(graph)

# Visualize the network
plot(graph, edge.width = E(graph)$weight * 10)

# Evaluate the network
cat("Number of Nodes:", vcount(graph), "\n")
cat("Number of Edges:", ecount(graph), "\n")
cat("Network Diameter:", diameter(graph), "\n")
