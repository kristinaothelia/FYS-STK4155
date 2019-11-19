import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestClassifier

#pd.DataFrame(X).fillna()

X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy', allow_pickle=True)

print(X)
print(y)


RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
RF.fit(X,y)

feature_importance = RF.feature_importances_
print(feature_importance)

plt.plot(feature_importance)
plt.show()