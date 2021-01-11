knn = KNeighborsClassifier(n_neighbors=3)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
knn.fit(X_train_pca, c_train)
print(f'Simple 3NN : {np.round(knn.score(X_test_pca, c_test),3)}')

trans_X_train = np.dot(X_train, A_rect_opt.T)
trans_X_test = np.dot(X_test, A_rect_opt.T)
knn.fit(trans_X_train, c_train)
print(f'3NN with NCA transformation : {np.round(knn.score(trans_X_test, c_test),3)}')