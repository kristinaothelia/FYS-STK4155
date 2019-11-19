import numpy as np

from sklearn.linear_model import LogisticRegression

X = np.load("features.npy", allow_pickle=True)
y = np.load("targets.npy", allow_pickle=True)
y = y.astype('int')
#y = y.reshape(len(y), 1)

A = X[85,:]
print(A)

solver = LogisticRegression()

for i in y:
    if i != 0 and i != 1:
        #print('yo')
        None

solver.fit(X, np.ravel(y))

solver.predict(X)

print(solver.score(X, y))
