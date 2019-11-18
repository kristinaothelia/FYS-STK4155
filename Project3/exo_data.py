import os, random, xlsxwriter

import seaborn               as sns
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt

cwd      = os.getcwd()
filename = cwd + '/cumulative_2019_all.xls'
nanDict  = {}
df       = pd.read_excel(filename, header=1, skiprows=85, index_col=0, na_values=nanDict)

# Check DONE/ACTIVE, if ACTIVE, drop this

# Removing the same columns as mentioned by the scientific article, do we want to drop more features or not?
df.drop(columns=['koi_longp', 'koi_ingress', 'koi_model_dof', 'koi_model_chisq', 'koi_sage'], axis=1, inplace=True)
df.drop(columns=['kepoi_name', 'koi_comment', 'koi_limbdark_mod', 'koi_parm_prov', 'koi_trans_mod'], axis=1, inplace=True)
df.drop(columns=['koi_datalink_dvr', 'koi_datalink_dvs', 'koi_pdisposition', 'kepler_name', 'koi_score'], axis=1, inplace=True)
df.drop(columns=['koi_time0bk', 'koi_tce_delivname', 'koi_sparprov', 'koi_vet_stat', 'koi_vet_date'], axis=1, inplace=True)
df.drop(columns=['koi_disp_prov', 'koi_ldm_coeff3', 'koi_ldm_coeff4'], axis=1, inplace=True)

print(df)


'''
# Dropping NaN values or not? If so, how??
# Asking Morten about this?
df = df.drop(df[(df.koi_dikco_mra  == 'NaN')].index)
df = df.drop(df[(df.koi_dikco_mdec == 'NaN')].index)
df = df.drop(df[(df.koi_dikco_msky == 'NaN')].index)
'''

def Corr_matrix():
	#Creating a correlation matrix for the dataframe
	# Why is this different from the paper? Is this a problem?
	sns.heatmap(df.corr())
	plt.title("Correlation Matrix")
	plt.tight_layout()
	plt.show()

#Corr_matrix()

# Locking the Confirmed, False Positive and Candidates (using this for plotting histogram of distribution?)
# The Candidates, later used for calculating the probability that a candidate is an exoplanet

df = df.drop(df[(df.koi_prad > 15)].index)		# TEST

CONFIRMED  = df.loc[df['koi_disposition']  == 'CONFIRMED']       # = 1
NEGATIVE   = df.loc[df['koi_disposition']  == 'FALSE POSITIVE']  # = 0
CANDIDATES = df.loc[df['koi_disposition']  == 'CANDIDATE']

#print(CONFIRMED)
#print(NEGATIVE)
#print(CANDIDATES)

def Histogram(feature_name, x_label):
	'''
	fig, ax = plt.subplots()
	labels, counts = np.unique(NEGATIVE.loc[:,  NEGATIVE.columns == feature_name].values, return_counts=True)
	plt.bar(labels, counts)
	labels2, counts2 = np.unique(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values, return_counts=True)
	plt.bar(labels2, counts2)
	labels3, counts3 = np.unique(CANDIDATES.loc[:,  CANDIDATES.columns == feature_name].values, return_counts=True)
	plt.bar(labels3, counts3)
	'''
	plt.hist(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values, alpha=0.8, bins=50, label="Confirmed")
	plt.hist(NEGATIVE.loc[:,   NEGATIVE.columns == feature_name].values, alpha=0.8, bins=100, label="False positive")
	plt.hist(CANDIDATES.loc[:, CANDIDATES.columns == feature_name].values, alpha=0.8, bins=100, label="Candidates")

	plt.title("Histogram for Kepler exoplanet data")
	plt.xlabel(x_label)
	plt.ylabel("Observations count")
	plt.xlim(0, 15)
	plt.legend()
	plt.show()

# Make histogram for planet radius [Earth radii], koi_prad
Histogram('koi_prad', "Planet radius [Earth radii], koi_prad")

# Creating DataFrame only with the Confirmed/False Positive (dropping the candidates)
# The features (not including the column koi_disposition or the Kepler ID)
# The targets, or the response variable, is the column koi_disposition (also not including the candidates)
CONFIRMED_NEGATIVE = df.drop(df[(df.koi_disposition == 'CANDIDATE')].index)
features = CONFIRMED_NEGATIVE.loc[:, (CONFIRMED_NEGATIVE.columns != 'koi_disposition') & (CONFIRMED_NEGATIVE.columns != "kepid")].values
target   = CONFIRMED_NEGATIVE.loc[:,  CONFIRMED_NEGATIVE.columns == 'koi_disposition'].values
print(features)
print(target)


# Renaming the targets to 0 and 1 instead of Confirmed/False Positives
target[target == 'CONFIRMED']      = 1
target[target == 'FALSE POSITIVE'] = 0

print(target)
