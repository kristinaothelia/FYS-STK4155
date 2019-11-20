import os, random, xlsxwriter

import seaborn               as sns
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

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
df.drop(columns=['koi_disp_prov', 'koi_ldm_coeff3', 'koi_ldm_coeff4', 'koi_fittype', 'koi_quarters'], axis=1, inplace=True)

df = df.replace(r'^\s*$', np.nan, regex=True)
df = pd.DataFrame.dropna(df, axis=0, how='any')


print(df)
header_names = list(df)
np.save('feature_names', header_names)


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


#df = df.drop(df[(df.koi_prad > 100)].index)
    
CONFIRMED  = df.loc[df['koi_disposition']  == 'CONFIRMED']       # = 1
NEGATIVE   = df.loc[df['koi_disposition']  == 'FALSE POSITIVE']  # = 0
CANDIDATES = df.loc[df['koi_disposition']  == 'CANDIDATE']

#print(CONFIRMED)
#print(NEGATIVE)
#print(CANDIDATES)

def Histogram2():
    g = df.loc[:, (df.columns == 'koi_disposition')].values
    labels, counts = np.unique(g, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.ylabel("Observations count")
    plt.show()

def Histogram(feature_name, x_label, title=None, logscale=False):

    
    if logscale:
        plt.hist(np.log(NEGATIVE.loc[:,   NEGATIVE.columns == feature_name].values), alpha=0.8, bins=10, label="False positive")
        plt.hist(np.log(CANDIDATES.loc[:, CANDIDATES.columns == feature_name].values), alpha=0.8, bins=10, label="Candidates")
        plt.hist(np.log(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values), alpha=0.8, bins=8, label="Confirmed")
        plt.xlim(-2, 15)
    else:        
        plt.hist(NEGATIVE.loc[:,   NEGATIVE.columns == feature_name].values, alpha=0.8, label="False positive")
        plt.hist(CANDIDATES.loc[:, CANDIDATES.columns == feature_name].values, alpha=0.8, label="Candidates")
        plt.hist(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values, alpha=0.8, label="Confirmed")
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Observations count")
    plt.legend()
    plt.show()
    

# Make histogram for planet radius [Earth radii], koi_prad
#Histogram('koi_prad', "log Planet radius [Earth radii]", title="Histogram for koi_prad", logscale=True)
#Histogram('koi_duration', "Transit duration [Hours]", title="Histogram for koi_duration")


neg = NEGATIVE.loc[:,   NEGATIVE.columns == 'koi_prad'].values
can = CANDIDATES.loc[:,   CANDIDATES.columns == 'koi_prad'].values
con = CONFIRMED.loc[:,   CONFIRMED.columns == 'koi_prad'].values

#print('hi')
#print(np.max(con))

"""
plt.plot(neg, 'ro')
plt.plot(can, 'go')
plt.plot(con, 'bo')
plt.show()
"""


# Creating DataFrame only with the Confirmed/False Positive (dropping the candidates)
# The features (not including the column koi_disposition or the Kepler ID)
# The targets, or the response variable, is the column koi_disposition (also not including the candidates)
CONFIRMED_NEGATIVE = df.drop(df[(df.koi_disposition == 'CANDIDATE')].index)
features = CONFIRMED_NEGATIVE.loc[:, (CONFIRMED_NEGATIVE.columns != 'koi_disposition') & (CONFIRMED_NEGATIVE.columns != "kepid")].values
target   = CONFIRMED_NEGATIVE.loc[:,  CONFIRMED_NEGATIVE.columns == 'koi_disposition'].values
#print(features)
#print(target)


# Renaming the targets to 0 and 1 instead of Confirmed/False Positives
target[target == 'CONFIRMED']      = 1
target[target == 'FALSE POSITIVE'] = 0

#print(target)

print('hi')
print(features[2,:])
scaler = StandardScaler() #RobustScaler() #MaxAbsScaler() #MinMaxScaler()
scaler.fit_transform(features)
print(features)

np.save('features', features)
np.save('targets', target)
np.save('candidates', CANDIDATES.loc[:, (CANDIDATES.columns != 'koi_disposition')].values)

