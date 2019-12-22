"""
FYS-STK4155 - Project 3:
A program that makes a scatter plot of predicted exoplanets from NASA's
CANDIDATE exoplanets. After being classified as (predicted) confirmed or false
positive exoplanet using a machine learning method,
AND defined that they lie within the Goldilock zone! (done in exo_data.py)

Input files:
GC	| 'Goldilock_PandasDataFrame.xlsx' 	| Confirmed exoplanets. Imported Pandas frame
GP	| 'Goldilock_predicted.npy'			| Predicted goldilock planets. Imported npy file
"""
import os
import pandas as pd
import numpy  as np
import matplotlib.pylab as plt

def GoldilocksZone(fil_, method, th):

	# import files
	cwd      = os.getcwd()
	filename = cwd + '/Goldilock_PandasDataFrame.xlsx'
	nanDict  = {}

	GC       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict) # df
	#GP       = np.load('Goldilock_predicted.npy', allow_pickle=True)
	GP       = fil_

	# List to store predicted 'False positives' and 'Confirmed' exoplanets
	P_FC     = []

	# Replace existing exoplanet list 'CANDIDATE' with predicted FP or C variable
	for i in range(len(GP)):
		if GP[i] == 0:
			P_FC.append('False Positives')
		else:
			P_FC.append('Confirmed')

	GC = GC.assign(koi_disposition=P_FC)

	inside   = GC.loc[GC['koi_disposition']  == 'Confirmed']
	outside  = GC.loc[GC['koi_disposition']  == 'False Positives']

	print(GC['koi_disposition']  == 'Confirmed')

	# Change threshold number to use in file name
	if th == 0.9:
		th_fil = 9
	elif th == 0.8:
		th_fil = 8
	elif th == 0.7:
		th_fil = 7
	elif th == 0.6:
		th_fil = 6
	elif th == 0.5:
		th_fil = 5

	# Make scatter plot of predicted planets in Goldilock zone
	#plt.style.use('dark_background')
	plt.plot(inside['koi_prad'], inside['koi_teq'], 'go', label='Predicted confirmed')
	plt.plot(outside['koi_prad'], outside['koi_teq'], 'm^', label='Predicted false positive') #m

	plt.title('Predicted planets in Goldilock zone \n %s and threshold=%g' %(method, th), fontsize=15)
	plt.xlabel('Planet radii [Earth radii]', fontsize=15)
	plt.ylabel('Planet surface temperature', fontsize=15)
	plt.legend(fontsize=15)
	plt.savefig('hab_exoplanets/scatter_%s_th%g.png' % (method, th_fil))
	plt.show()
