from sklearn.neighbors import NeighborhoodComponentsAnalysis
nca = NeighborhoodComponentsAnalysis()
nca.fit(X_train, c_train)

print('On full data :')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, c_train)

print(f'KNN performance on base data : {np.round(knn.score(X_test, c_test),3)}')

knn.fit(nca.transform(X_train), c_train)

print(f'KNN performance on NCA transformed data : {np.round(knn.score(nca.transform(X_test), c_test),3)}')

nca = NeighborhoodComponentsAnalysis(n_components=2)
nca.fit(X_train, c_train)

print('On dimension-reduced data :')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, c_train)

print(f'KNN performance on PCA transformed data : {np.round(knn.score(X_test_pca, c_test),3)}')

knn.fit(nca.transform(X_train), c_train)

print(f'KNN performance on NCA transformed data : {np.round(knn.score(nca.transform(X_test), c_test),3)}')