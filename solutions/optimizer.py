from scipy.optimize import minimize
optimizer_params = {'method': 'L-BFGS-B',
                    'fun': grad,
                    'args': (X_train, same_class_mask),
                    'jac': True,
                    'x0': A,
                    'tol': 1e-5,
                    'options': dict(maxiter=50, disp=-1),
                    'callback': None
                    }

# Call the optimizer
opt_result = minimize(**optimizer_params)
A_opt = opt_result.x.reshape(-1, X_train.shape[1])