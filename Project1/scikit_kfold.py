#plt.figure(2)
		#plt.plot(complexity, MSE_test_check, label='testing')
		#plt.plot(complexity, MSE_train_check, label='training')
		#plt.plot(complexity, error)
		#plt.legend()
		#plt.show()



from sklearn.model_selection import KFold
		from sklearn.pipeline import Pipeline
		from sklearn.pipeline import make_pipeline
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression
		X = project1_func.CreateDesignMatrix(x,y, n=p_degree)
		k=5
		X_trainz, X_testz, y_trainz, y_testz = train_test_split(X,data,test_size=1./k)
		array_size_thingy=len(y_testz)

		kf = KFold(n_splits=k) 
		kf.get_n_splits(X)
		kfold = KFold(n_splits = k,shuffle=True,random_state=5)
		err = []
		#bi=[]
		#vari=[]




#X_train, X_test, data_train, data_test = train_test_split(X, data, shuffle=True, test_size=0.3, random_state=None)
			#betas = project1_func.beta(data_train, X_train, method='OLS')

			'''
			from sklearn.model_selection import KFold # import KFold

			#kf = KFold(n_splits=5) # Define the split - into 10 folds 
			#kf.get_n_splits(X)	    # returns the number of splitting iterations in the cross-validator
			#KFold(n_splits=5, random_state=5, shuffle=True)

			y_pred = np.empty((array_size_thingy, k))
			j=0
			model = make_pipeline(PolynomialFeatures(degree=degree),LinearRegression(fit_intercept=False))

			for train_inds, test_inds in kf.split(X):
				#X_train, X_test = X[train_index], X[test_index]
				#data_train, data_test = data[train_index], data[test_index]
				xtrain = X[train_inds]
				ytrain= data[train_inds]
				xtest = X[test_inds]
				ytest = data[test_inds]
				y_pred[:,j] = model.fit(xtrain,ytrain).predict(xtest).ravel()
				j+=1
			#ytest[:,j] = ytest.ravel()
			#print(y_pred[:,0].shape)
			#print(ytest.shape)
			#print(y_pred.shape)
			error = np.mean( np.mean((ytest - y_pred[:,0])**2) )
			#bias = np.mean( (ytest - np.mean(y_pred, axis=1, keepdims=True))**2 )
			#variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
			err.append(error)
			#bi.append(bias)
			#vari.append(variance)
			'''
			

			'''
			betas = project1_func.beta(data_train, X_train, method='OLS')
			#ytilde = X_train @ betas
			MSE_train_check[degree] = project1_func.MeanSquaredError(data_train,ytilde)
			#ypredict = X_test @ betas
			MSE_test_check[degree] = project1_func.MeanSquaredError(data_test,ypredict)
			'''