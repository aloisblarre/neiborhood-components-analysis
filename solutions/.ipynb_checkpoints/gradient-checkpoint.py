weighted_p_ij = masked_p_ij - p_ij * p
weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))
gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(X_train)