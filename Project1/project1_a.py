from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	#noise = np.random.normal(size=len(x))
	#noise = np.random.random(len(x)) * 1
	return term1 + term2 + term3 + term4 #+ noise

def CreateDesignMatrix(x, y, n):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X


def MeanSquaredError(y_data, y_model):
	n = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)
	return MSE


def ScoreFunction(y_data, y_model):
	counter = np.sum((y_data-y_model)**2)
	denominator = np.sum((y_data-np.mean(y_data))**2)
	R_2 = 1 - (counter/denominator)
	return R_2

def RelativeError(y_data, y_model):
	error = abs((y_data-y_model)/y_data)
	return error

def OrdinaryLeastSquares(data, X):
	betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
	OLS = np.dot(X, betas)
	return OLS

def Plotting(x, y, z, model):
	plot_data = np.reshape(z, (len(x), len(y)))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(np.ravel(x), np.ravel(y), model, s=5, c='g', marker='o', alpha=1.0)
	#surf = ax.plot_surface(x, y, plot_data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	# Customize the z axis.
	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# Add a color bar which maps values to colors.
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()


n_x = 101  # number of points 
m  	= 5    # degree of polynomial 

# Creating data values x and y 
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z 	 = FrankeFunction(x,y)

#Transform from matrices to vectors
x_vec=np.ravel(x)
y_vec=np.ravel(y)
n=int(len(x_vec))
data=np.ravel(z) #+ np.random.random(n) * 1



# create the design matrix
X = CreateDesignMatrix(x,y, n=5)

model = OrdinaryLeastSquares(data, X)

MSE = MeanSquaredError(data, model)
print(MSE)

R_2 = ScoreFunction(data, model)
print(R_2)


Plotting(x, y, data, model)





#fit = np.linalg.lstsq(X, Energies, rcond =None)[0]
#ytilde = np.dot(fit,X.T)


'''
degree = 5
weights = np.polyfit(x, data, degree)
p = np.poly1d(weights)
model = np.polyval(p, x)
#print(result)

mse = MeanSquaredError(data, model)
score = ScoreFunction(data, model)

print(mse)
print(score)
'''


'''
x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)

p = np.poly1d(np.polyfit(x, y,2))
print(p)

plt.plot(x, y, '-')
plt.show()
'''




def k_fold(data, X, k):
	"""
	Taking in:
	Returning:
	"""

	print(X)
	data_split = np.array(np.array_split(data, k, axis=0)) # splitting in k-folds
	X_split = np.array_split(X, k, axis=0)                 # splitting in k-folds

	X_split = np.array(X_split) #.reshape(-1,1)  #.squeeze()

	MSE_error = list() # list for storing the MSE values

	for index in range(k):

		testset_data = data_split[index]                 # the index row 
		trainset_data = np.delete(data_split, index, 0)  # all except the index row 

		
		testset_X = X_split[index].astype(np.float64) 
		trainset_X = np.delete(X_split, index, 0)

		print(testset_X)
		print(trainset_X)

		'''
		betas_k = beta(trainset_data, trainset_X)

		ytilde_k = trainset_X @ betas_k

		train_MSE = MeanSquaredError(trainset_data,ytilde_k)

		ypredict_k = testset_X @ betas_k

		test_MSE = MeanSquaredError(testset_data,ypredict_k)

		MSE_error.append(test_MSE)
		'''
		
	return MSE_error
