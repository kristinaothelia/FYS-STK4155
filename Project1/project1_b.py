import matplotlib.pyplot as plt

#from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

#import sklearn
#print(sklearn.__version__)



# y_model = y_tilde


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	noise = np.random.normal(size=len(x))
	return term1 + term2 + term3 + term4 # + noise

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



n_x = 1000
m = 5 # degree of polynomial 

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

#Transform from matrices to vectors
x_vec=np.ravel(x)
y_vec=np.ravel(y)
n=int(len(x_vec))
data=np.ravel(z) #+ np.random.random(n) * 1


# finally create the design matrix
X = CreateDesignMatrix(x,y, n=5)

X_train, X_test, data_train, data_test = train_test_split(X, data, shuffle=False, test_size=0.2)



#clf = skl.LinearRegression().fit(X_train, data_train)
#ytilde = clf.predict(X_train, data_train)

#linreg = skl.LinearRegression()
#linreg.fit(x,y)

'''
def k_fold(data_set, k, randomize=False):
	# k = 3 (small dataset) number of folds
	# k = 10 (larger dataset) number of folds

	#if randomize:
    #    items = list(items)
    #    shuffle(items)

	dataset_split = list()          # storing train data
	dataset_copy = list(data_set)   # test data 
	fold_size = int(len(data_set)/k)
	for i in range(k):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	train_set = np.array(dataset_split) # returns train set ??
	test_set = np.array(dataset_copy) # returns test set ??
	return test_set, train_set
'''


'''
def k_fold(data_set, k, randomize=False):
	"""
	Taking in:
	Returning:
	"""
	
	# k = 3 (small dataset) number of folds
	# k = 10 (larger dataset) number of folds

	#if randomize:
    #    data_set = list(data_set)
    #    shuffle(data_set)

	train_set = list()               # storing train data
	test_dataset = list()
	dataset_copy = list(data_set)    # copy of the dataset
	fold_size = int(len(data_set)/k) 
	for i in range(k):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy)) # picks a random 'index'
			fold.append(dataset_copy.pop(index)) # adds the randomly picked data to fold
		
		train_set.append(fold)                   # appends fold to train
		test_dataset.append(dataset_copy)        # appends remaining data to test 

		train_set = np.array(train_set)          # returns train set ??
		test_set  = np.array(dataset_copy)       # returns test set ??

		# Can now use train and test in OLS for each k ?
		
	return test_set, train_set
'''