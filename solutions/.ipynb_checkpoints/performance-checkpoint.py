from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)
print(f'Simple 3NN : {np.round(knn.score(X_test, c_test),3)}')

trans_X_train = np.dot(X_train, A_opt.T)
trans_X_test = np.dot(X_test, A_opt.T)
knn.fit(trans_X_train, c_train)
print(f'3NN with NCA transformation : {np.round(knn.score(trans_X_test, c_test),3)}')