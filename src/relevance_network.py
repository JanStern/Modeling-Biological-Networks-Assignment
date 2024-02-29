import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def compute_mutual_info_matrix(df):
    n = df.shape[1]
    mi_matrix = pd.DataFrame(np.zeros((n, n)), columns=df.columns, index=df.columns)
    for i, column1 in enumerate(df.columns):
        for j, column2 in enumerate(df.columns[i:], i):
            if i == j:
                mi_matrix.at[column1, column2] = mutual_info_regression(df[[column1]], df[column2])
            else:
                mi_matrix.at[column1, column2] = mutual_info_regression(df[[column1]], df[column2])
                mi_matrix.at[column2, column1] = mi_matrix.at[column1, column2]
    return mi_matrix


def apply_aracne(mi_matrix, threshold, dpi_threshold):
    """
    Apply the Data Processing Inequality (DPI) to a mutual information matrix.

    Parameters:
    - mi_matrix: 2D numpy array, mutual information matrix.
    - threshold: float, threshold for considering an interaction significant.
    - dpi_threshold: float, threshold for the DPI step.

    Returns:
    - A numpy array with the same shape as mi_matrix, where indirect interactions have been removed.
    """
    # Number of genes
    n_genes = mi_matrix.shape[0]

    # Create a copy of the MI matrix to modify
    # filtered_mi_matrix = np.copy(mi_matrix)
    filtered_mi_matrix = (mi_matrix).copy()

    # Apply threshold
    filtered_mi_matrix[filtered_mi_matrix < threshold] = 0

    # Apply DPI
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            for k in range(n_genes):
                if k != i and k != j:
                    # If the interaction between i and j is weaker than both interactions i-k and j-k,
                    # it's considered an indirect interaction and removed
                    if min(filtered_mi_matrix.iloc[i, j], dpi_threshold) < min(filtered_mi_matrix.iloc[i, k],
                                                                          filtered_mi_matrix.iloc[j, k]):
                        filtered_mi_matrix.iloc[i, j] = 0
                        filtered_mi_matrix.iloc[j, i] = 0

    return filtered_mi_matrix
