import matplotlib.pyplot as plt
import numpy as np






def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def MeanSquaredError(y, y_tilde, n):

	MSE = (1/n)*sum((y-y_tilde)**2)
	return MSE

def MeanValue(y, n):
	mean_value = (1/n)*sum(y)
	#print(mean_value, np.mean(y)) can we just use mean?
	return mean_value

def ScoreFunction(y, y_tilde, n):
	y_mean = MeanValue(y, n)
	counter = n*MeanSquaredError(y, y_tilde, n)
	denominator = sum((y-y_mean)**2)
	R_2 = 1 - (counter/denominator)
	return R_2


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

n = len(x)

noise = np.random.normal(size=n)
print(noise)

franke = FrankeFunction(x,y)
mse = MeanSquaredError(y, 0.1, n)
score = ScoreFunction(y, 0.1, n)