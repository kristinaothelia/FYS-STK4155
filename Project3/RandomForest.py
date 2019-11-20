import os
import numpy 			 as np
import matplotlib.pyplot as plt
import scikitplot        as skplt
import functions         as func

from sklearn.impute 		 import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble 		 import RandomForestClassifier


#from sklearn import tree
#pd.DataFrame(X).fillna()
# grid search 

TrainingShare = 0.70
seed 		  = 0

X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy', allow_pickle=True)
candidates   = np.load('candidates.npy', allow_pickle=True)
header_names = np.load('feature_names.npy', allow_pickle=True)
feature_list = header_names[1:]

print(X)
print(y)

y = y.astype('int')
y = np.ravel(y)

print(len(y))
#print(sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainingShare, test_size = 1-TrainingShare, random_state=seed)


# plot error against number of trees
RF = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed, criterion='gini')
RF.fit(X_train,y_train)

predict     = RF.predict(X_test)
accuracy 	= RF.score(X_test,y_test)

precision   = func.precision(y_test, predict)
recall      = func.recall(y_test, predict)
f1_score    = func.F1_score(y_test, predict)

# Calculate the absolute errors
errors = abs(predict - y_test)

# Calculating the different metrics:
print('\n-------------------------------------------')
print('The accuracy is    : %.3f' % accuracy)
print('The F1 score is    : %.3f' % f1_score)
print('The precision is   : %.3f' % precision)
print('The recall is      : %.3f' % recall)
print('The absolute error : %.3f' % np.mean(errors))
print('-------------------------------------------')


skplt.metrics.plot_confusion_matrix(y_test, predict)
plt.show()


#print(RF.decision_path(X_test))

# Pull out one tree from the forest
tree_number = 5
tree 		= RF.estimators_[tree_number]
func.PlotOneTree(tree, feature_list)

predict_candidates       = np.array(RF.predict(candidates))

predicted_false_positive = (predict_candidates == 0).sum()
predicted_exoplanets     = (predict_candidates == 1).sum()


print('')
print('The Random Forest Classifier predicted')
print('--------------------------------------')
print('%g exoplanets      of %g candidates'  %(predicted_exoplanets, len(predict_candidates)))
print('%g false positives  of %g candidates'  %(predicted_false_positive, len(predict_candidates)))


'''
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


'''
from matplotlib.ticker import MaxNLocator
plt.barh(feature_importance, width=0.1,  align='center')
plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
plt.show()
'''
