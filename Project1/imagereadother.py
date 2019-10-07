from PIL import Image
		terreng_image = Image.open('Oslo.tif')
		print(terreng_image.size)
		terreng_image.show()


for power in range(maxpower):
    poly2 = PolynomialFeatures(degree=power+1)
    Lasso_sklearn = poly2.fit_transform(lasso_multi)
    print(power)
    for lamb in range(len(lambdas)): 
        errin=0
        errout=0
        r2in=0
        r2out=0
        for train, test in kf.split(X):
            x_train = X[train]
            y_train = frank[train]
            x_test = X[test]
            y_test = frank[test]
            
            
            Lasso_train = poly2.fit_transform(x_train)
            Lasso_test =poly2.fit_transform(x_test)
            lasso = linear_model.Lasso(alpha=lambdas[lamb], fit_intercept=False, max_iter=10e5)
            fit_lasso = lasso.fit(Lasso_train, y_train) 
            predictLasso = lasso.predict(Lasso_test)
            predictLasso_train = lasso.predict(Lasso_train)
            errin += fx.MSE(predictLasso_train,y_train)
            errout += fx.MSE(predictLasso,y_test)
            r2in += fx.R2Score(predictLasso_train,y_train)
            r2out += fx.R2Score(predictLasso,y_test)
        mse_lasso_in[power,lamb] = errin/k
        mse_lasso_out[power,lamb] = errout/k
        r2_lasso_in[power,lamb] = r2in/k
        r2_lasso_out[power,lamb] = r2out/k




print(terrain_arr.shape, X.shape)
		model = project1_func.OrdinaryLeastSquares(terrain_arr, X)

		MSE = project1_func.MeanSquaredError(terrain_arr, model)
		R_2 = project1_func.R2_ScoreFunction(terrain_arr, model)
		print(MSE)
		print(R_2)

		plot_model = np.reshape(model, (len(x[:,0]), len(y[0,:])))

		plt.figure(2)
		plt.title('Terrain over Norway 1')
		plt.imshow(plot_model, cmap='gray')
		plt.colorbar(shrink=0.5, aspect=5)
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()
