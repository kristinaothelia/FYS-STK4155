"""
FYS-STK4155 - Project 3:
A program that
"""
import os, random, xlsxwriter
import seaborn             as sns
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import functions           as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
#------------------------------------------------------------------------------
cwd      = os.getcwd()
filename = cwd + '/cumulative_2019_all.xls'
nanDict  = {}
df       = pd.read_excel(filename, header=1, skiprows=85, index_col=0, na_values=nanDict)

# Removing the same columns as mentioned by the scientific article, do we want to drop more features or not?
df.drop(columns=['koi_longp', 'koi_ingress', 'koi_model_dof', 'koi_model_chisq', 'koi_sage'], axis=1, inplace=True)
df.drop(columns=['kepoi_name', 'koi_comment', 'koi_limbdark_mod', 'koi_parm_prov', 'koi_trans_mod'], axis=1, inplace=True)
df.drop(columns=['koi_datalink_dvr', 'koi_datalink_dvs', 'koi_pdisposition', 'kepler_name', 'koi_score'], axis=1, inplace=True)
df.drop(columns=['koi_time0bk', 'koi_tce_delivname', 'koi_sparprov', 'koi_vet_stat', 'koi_vet_date'], axis=1, inplace=True)
df.drop(columns=['koi_disp_prov', 'koi_ldm_coeff3', 'koi_ldm_coeff4', 'koi_fittype', 'koi_quarters'], axis=1, inplace=True)

df = df.replace(r'^\s*$', np.nan, regex=True)
df = pd.DataFrame.dropna(df, axis=0, how='any')

header_names = list(df)
np.save('feature_names', header_names)


def Corr_matrix():
	""" Creating a correlation matrix for the dataframe """
	sns.heatmap(df.corr())
	plt.title("Correlation Matrix")
	plt.tight_layout()
	plt.show()

Corr_matrix()

# Locking the Confirmed, False Positive and Candidates (using this for plotting histogram of distribution?)
# The Candidates, later used for calculating the probability that a candidate is an exoplanet

CONFIRMED  = df.loc[df['koi_disposition']  == 'CONFIRMED']       # = 1
NEGATIVE   = df.loc[df['koi_disposition']  == 'FALSE POSITIVE']  # = 0
CANDIDATES = df.loc[df['koi_disposition']  == 'CANDIDATE']

# Make histograms
def Histogram(feature_name, x_label, title=None, logscale=False):

    if logscale:
        plt.hist(np.log(NEGATIVE.loc[:,   NEGATIVE.columns == feature_name].values),   alpha=0.8, bins=10, label="False positive")
        plt.hist(np.log(CANDIDATES.loc[:, CANDIDATES.columns == feature_name].values), alpha=0.8, bins=10, label="Candidates")
        plt.hist(np.log(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values),  alpha=0.8, bins=8, label="Confirmed")
        plt.xlim(-2, 15)
    else:
        plt.hist(NEGATIVE.loc[:,   NEGATIVE.columns == feature_name].values,   alpha=0.8, label="False positive")
        plt.hist(CANDIDATES.loc[:, CANDIDATES.columns == feature_name].values, alpha=0.8, label="Candidates")
        plt.hist(CONFIRMED.loc[:,  CONFIRMED.columns == feature_name].values,  alpha=0.8, label="Confirmed")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Observations count")
    plt.legend()
    plt.show()

#Histogram('koi_prad', "log Planet radius [Earth radii]", title="Histogram for koi_prad", logscale=True)
#Histogram('koi_duration', "Transit duration [Hours]", title="Histogram for koi_duration")

# Make histogram of candidates, confirmed and false positive objects
F.HistogramKOI(g=df.loc[:, (df.columns == 'koi_disposition')].values)

neg = NEGATIVE.loc[:,   NEGATIVE.columns   == 'koi_prad'].values
can = CANDIDATES.loc[:, CANDIDATES.columns == 'koi_prad'].values
con = CONFIRMED.loc[:,  CONFIRMED.columns  == 'koi_prad'].values

# Creating DataFrame only with the Confirmed/False Positive (dropping the candidates)
# The features (not including the column koi_disposition or the Kepler ID)
# The targets, or the response variable, is the column koi_disposition (also not including the candidates)
CONFIRMED_NEGATIVE = df.drop(df[(df.koi_disposition == 'CANDIDATE')].index)
features = CONFIRMED_NEGATIVE.loc[:, (CONFIRMED_NEGATIVE.columns != 'koi_disposition') & (CONFIRMED_NEGATIVE.columns != "kepid")].values
target   = CONFIRMED_NEGATIVE.loc[:,  CONFIRMED_NEGATIVE.columns == 'koi_disposition'].values

# Renaming the targets to 0 and 1 instead of Confirmed/False Positives
target[target == 'CONFIRMED']      = 1
target[target == 'FALSE POSITIVE'] = 0

scaler = StandardScaler() #RobustScaler() #MaxAbsScaler() #MinMaxScaler()
scaler.fit_transform(features)

def GoldiLock_Candidates(temp_max=390, temp_min=260, rad_max=2.5, rad_min=0.5):
	"""
	Default values for goldielock zone:
	Exoplanet surface temperature max [K]	| 390
	Exoplanet surface temparature min [K]	| 260
	Exoplanet radius max [Earth radii]		| 2.5
	Exoplanet radius min [Earth radii]		| 0.5
	"""
	GoldiLock = CANDIDATES.loc[CANDIDATES['koi_teq'] < temp_max]
	GoldiLock = GoldiLock.loc[GoldiLock['koi_teq']   > temp_min]
	GoldiLock = GoldiLock.loc[GoldiLock['koi_prad']  < rad_max]
	GoldiLock = GoldiLock.loc[GoldiLock['koi_prad']  > rad_min]
	return GoldiLock

# Find predicted exoplanets in goldilock zone, by temperature and radius.
GoldiLocks = GoldiLock_Candidates(323, 273, 2.5, 0.5)

# Save files for use in other programs
np.save('features', features)
np.save('targets', target)
np.save('candidates', CANDIDATES.loc[:, (CANDIDATES.columns != 'koi_disposition')].values)
np.save('GoldiLock', GoldiLocks.loc[:, (GoldiLocks.columns != 'koi_disposition')].values)

# Save pands DataFrame of predicted exoplanets within the goldielock zone
GoldiLocks.to_excel('GoldiLock_PandasDataFrame.xlsx')
