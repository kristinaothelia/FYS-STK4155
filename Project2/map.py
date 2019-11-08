import matplotlib.pyplot as plt
import numpy as np

arr = np.load('acc_score.npy', allow_pickle=True)

new_arr = np.zeros(shape=arr.shape)

Ni = len(arr[:,0])
Nj = len(arr[0,:])

for i in range(Ni):
    for j in range(Nj):
        new_arr[i][j] = arr[i][j]

plt.imshow(new_arr, origin="lower")
plt.colorbar()
plt.ylabel('eta')
plt.xlabel('lambda')
plt.show()

