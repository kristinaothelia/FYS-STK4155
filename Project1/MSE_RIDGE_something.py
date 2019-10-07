		for l in range(0, len(lambdas)):
			#print('for lambda=%g, calculates ', %lambdas[l])

			train_MSE = np.zeros(p_degree)
			test_MSE = np.zeros(p_degree)
			bias = np.zeros(p_degree)
			variance = np.zeros(p_degree)

			for degree in range(0, p_degree):
				X = project1_func.CreateDesignMatrix(x,y, n=degree)
				train_MSE[degree], test_MSE[degree], bias[degree], variance[degree] = project1_func.k_fold(data, X, 10, method='Ridge', l=lambdas[l])

			complexity = np.arange(0,p_degree)
			grid[l, degree] = test_MSE[degree]
			#plt.plot(complexity, test_MSE, label='%g' %lambdas[l])
			
				#MSE_test[degree] =	 np.min(test_MSE)   # adding the minimum?
				#best_lambda[degree] = lambdas[np.argmin(test_MSE)]
			#print('for lambda=%g, degree=%g gives the lowest MSE' %(lambdas[l], degree))

			#print(best_lambda[degree])
			#MSE_train[degree] =  np.min(train_MSE_list)  # adding the minimum?
			#bias[degree] = 		 np.min(bias_list)		 # adding the minimum?
			#variance[degree] =   np.min(variance_list)   # adding the minimum?

			#print(p_degree)
			#print(MSE_test[degree])


		#plt.figure(2)
		#plt.plot(complexity, MSE_test, label='test')
		#plt.plot(complexity, MSE_train, label='train')
		#plt.plot(complexity, bias, label='bias')
		#plt.plot(complexity, variance, label='variance')
		#plt.title('Ridge')
		#plt.legend()

		#plt.imshow(grid, origin='lower')
		#plt.legend()
		plt.show()