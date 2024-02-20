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

# Apply Data Processing Inequality (DPI) to eliminate indirect interactions
# Set DPI tolerance threshold
dpi_threshold <- 0.01 # Adjust based on your understanding of the data

# Function to apply DPI and remove indirect interactions
apply_dpi <- function(mi_matrix, threshold) {
  n <- ncol(mi_matrix)
  
  # Iterate over all triplets of genes
  for (i in 1:(n-2)) {
    for (j in (i+1):(n-1)) {
      for (k in (j+1):n) {
        # Find the minimum MI in each triplet
        min_mi <- min(mi_matrix[i, j], mi_matrix[i, k], mi_matrix[j, k])
        
        # If the minimum MI is below the DPI threshold, set it to 0
        if (min_mi < threshold) {
          # Identify the pair with the minimum MI
          if (min_mi == mi_matrix[i, j]) {
            mi_matrix[i, j] <- 0
            mi_matrix[j, i] <- 0
          } else if (min_mi == mi_matrix[i, k]) {
            mi_matrix[i, k] <- 0
            mi_matrix[k, i] <- 0
          } else {
            mi_matrix[j, k] <- 0
            mi_matrix[k, j] <- 0
          }
        }
      }
    }
  }
  
  return(mi_matrix)
}


mi_matrix <- calc_mutual_information(discretized_data)
print("MI Matrix")
print(mi_matrix)

# Apply DPI to the MI matrix
mi_matrix_dpi <- apply_dpi(mi_matrix, dpi_threshold)
print("MI ARACNE Matrix")
print(mi_matrix_dpi)


# Filter connections based on a threshold and create a network
threshold <- 0.03 # Set your mutual information threshold here
graph <- graph_from_adjacency_matrix(mi_matrix_dpi > threshold, weighted = TRUE)
graph <- simplify(graph)

# Visualize the network
plot(graph, edge.width = E(graph)$weight * 10)

# Evaluate the network
cat("Number of Nodes:", vcount(graph), "\n")
cat("Number of Edges:", ecount(graph), "\n")
cat("Network Diameter:", diameter(graph), "\n")
