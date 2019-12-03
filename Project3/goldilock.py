import os
import pandas as pd
import numpy  as np
import matplotlib.pylab as plt

cwd      = os.getcwd()
filename = cwd + '/Goldilock_PandasDataFrame.xlsx'
nanDict  = {}
df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)


Goldilocks_predicted = np.load('Goldilock_predicted.npy', allow_pickle=True)


t = []

for i in range(len(Goldilocks_predicted)):
	if Goldilocks_predicted[i] == 0:
		t.append('False Positives')
	else:
		t.append('Confirmed')


df = df.assign(koi_disposition=t)
print(df)

inside   = df.loc[df['koi_disposition']  == 'Confirmed']
outside  = df.loc[df['koi_disposition']  == 'False Positives']


#plt.style.use('dark_background')
plt.plot(inside['koi_prad'], inside['koi_teq'], 'go', label='Predicted confirmed')
plt.plot(outside['koi_prad'], outside['koi_teq'], 'm^', label='Predicted false positive') #m
plt.legend(fontsize=15)
plt.title('Predicted planets in Goldilock zone', fontsize=15)
plt.xlabel('Planet radii [Earth radii]', fontsize=15)
plt.ylabel('Planet surface temperature', fontsize=15)
plt.show()