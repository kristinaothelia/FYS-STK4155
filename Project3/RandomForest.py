import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestClassifier

#pd.DataFrame(X).fillna()

X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy', allow_pickle=True)

print(X)
print(y)

y=y.astype('int')


RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
RF.fit(X,y)
RF.predict(X)
acc = RF.score(X,y)
print(acc)

feature_importance = RF.feature_importances_
print(feature_importance)
print(len(feature_importance))


#for i in range(len(feature_importance)):
	# Check the i in feature_importance 
	# assign corresponding header name


plt.hist(feature_importance, align='left', histtype='bar', orientation='horizontal', rwidth=0.3)
plt.title('Feature Importance')
plt.xlabel('--')
plt.ylabel('--')
#plt.xlim([lb-width/2, ub-width/2])
plt.show()


'''
from matplotlib.ticker import MaxNLocator
plt.barh(feature_importance, width=0.1,  align='center')
plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
plt.show()
'''