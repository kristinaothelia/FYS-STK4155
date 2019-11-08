import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

arr     = np.load('acc_score.npy', allow_pickle=True)
etas    = np.load('eta_values.npy', allow_pickle=True)
lambdas = np.load('lambda_values.npy', allow_pickle=True)

new_arr = np.zeros(shape=arr.shape)

Ni = len(arr[:,0])
Nj = len(arr[0,:])

for i in range(Ni):
    for j in range(Nj):
        new_arr[i][j] = arr[i][j]

ax = sns.heatmap(new_arr, xticklabels=lambdas, yticklabels=etas, annot=True, linewidths=.3)
plt.title('Accuracy')
plt.ylabel('eta')
plt.xlabel('lambda')
plt.show()


#plt.style.use('dark_background')
#plt.imshow(new_arr, origin="lower")
#plt.colorbar()
#plt.ylabel('eta')
#plt.xlabel('lambda')
#plt.show()
