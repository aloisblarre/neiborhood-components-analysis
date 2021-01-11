def grad(A, x, same_class_mask):
    A = A.reshape(-1, x.shape[1])
    X_embedded = np.dot(x, A.T) 
    p_ij = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(p_ij, np.inf)
    p_ij = softmax(-p_ij)
    masked_p_ij = p_ij * same_class_mask
    p = np.sum(masked_p_ij, axis=1, keepdims=True)
    loss = np.sum(p)
    weighted_p_ij = masked_p_ij - p_ij * p
    weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
    np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))
    gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(x)
    return - loss, - gradient.ravel()