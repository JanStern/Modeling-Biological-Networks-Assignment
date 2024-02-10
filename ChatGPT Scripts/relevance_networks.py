import numpy as np
import matplotlib.pyplot as plt

from src.commons.data_transformation import read_data


# Step 2: Choose a Measure of Relevance - Pearson correlation coefficient
def calculate_pearson_correlation(gene1, gene2):
    """
    Calculate the Pearson correlation coefficient between two genes.
    """
    correlation, _ = np.corrcoef(gene1, gene2)
    return correlation


# Step 3: Calculate Pairwise Measures
def construct_relevance_matrix(gene_data):
    num_genes = gene_data.shape[1]
    relevance_matrix = np.zeros((num_genes, num_genes))

    for i in range(num_genes):
        for j in range(i, num_genes):
            if i != j:
                correlation = calculate_pearson_correlation(gene_data[:, i], gene_data[:, j])
                relevance_matrix[i, j] = correlation
                relevance_matrix[j, i] = correlation  # Symmetric matrix
            else:
                relevance_matrix[i, j] = 1  # Self-correlation is always 1
    return relevance_matrix


# Step 4 & 5: Thresholding and Network Construction
# For simplicity, we're directly using the relevance matrix as the network.
# In practice, you'd apply a threshold here to decide significant correlations.
def construct_network_from_relevance_matrix(relevance_matrix):
    # Apply threshold here if needed
    return relevance_matrix  # For demonstration, the relevance matrix itself is treated as the network


# Putting it all together
def main(gene_data):
    relevance_matrix = construct_relevance_matrix(gene_data)
    network = construct_network_from_relevance_matrix(relevance_matrix)
    print("Relevance Matrix (Network):\n", network)


t, data_gen_exp_t = read_data('../data/data_original.json')
data_gen_exp_t = np.array(list(data_gen_exp_t.values()))

# Example gene_data (replace this with your actual data)
main(data_gen_exp_t.T)
