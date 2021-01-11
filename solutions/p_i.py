same_class_mask = c_train[:, np.newaxis] == c_train[np.newaxis, :]
masked_p_ij = p_ij * same_class_mask
p = np.sum(masked_p_ij, axis=1, keepdims=True)