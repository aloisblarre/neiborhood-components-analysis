from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import softmax

X_embedded = np.dot(X_train, A.T) 
p_ij = pairwise_distances(X_embedded, squared=True)
np.fill_diagonal(p_ij, np.inf)
p_ij = softmax(-p_ij)