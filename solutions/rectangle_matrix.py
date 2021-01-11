from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
A_rect = pca.components_

optimizer_params = {'method': 'L-BFGS-B',
                    'fun': grad,
                    'args': (X_train, same_class_mask),
                    'jac': True,
                    'x0': A_rect,
                    'tol': 1e-5,
                    'options': dict(maxiter=50, disp=-1),
                    'callback': None
                    }

# Call the optimizer
opt_result = minimize(**optimizer_params)
A_rect_opt = opt_result.x.reshape(-1, X_train.shape[1])