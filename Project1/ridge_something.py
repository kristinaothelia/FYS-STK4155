	elif Ridge_method == True:
		print('Part d: Ridge Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		n_x = 500      # number of points 
		p_degree = 10  # degree of polynomial 

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		data = project1_func.Create_data(x, y, z, noise=True)


		lambdas = np.linspace(-3, 3, 9)
		print(lambdas)

		MSE_test = np.zeros(len(lambdas))
		MSE_train = np.zeros(len(lambdas))
		bias_l = np.zeros(len(lambdas))
		variance_l = np.zeros(len(lambdas))

		for l in range(0, len(lambdas)):

			test_MSE_list = list()
			train_MSE_list = list()
			bias_list = list()
			variance_list = list()

			for degree in range(0, p_degree):

				X = project1_func.CreateDesignMatrix(x,y, n=degree)

				train_MSE, test_MSE, bias, variance = \
				project1_func.k_fold(data, X, 10, method='Ridge', l=lambdas[l])

				train_MSE_list.append(train_MSE)
				test_MSE_list.append(test_MSE)
				bias_list.append(bias)
				variance_list.append(variance)


			MSE_test[l] = np.mean(test_MSE_list)   # adding the minimum?
			MSE_train[l] = np.mean(train_MSE_list) # adding the minimum?

		print(MSE_test)
		print(MSE_train)

		plt.plot(lambdas, test_MSE)
		plt.plot(lambdas, train_MSE)

		plt.title('Ridge')
		plt.show()