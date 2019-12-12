"""
FYS-STK4155 - Project 3: XGBoost
"""
import numpy   as np
import xgboost as xgb

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score
#------------------------------------------------------------------------------

X = np.load("features.npy", allow_pickle=True)
y = np.load("targets.npy", allow_pickle=True)
y = y.astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test  = xgb.DMatrix(X_test, label=Y_test)

param      = {  "eta": 0.3,
                "max_depth": 3,
                "objective": "multi:softprob",
                "num_class": 2  }
steps      = 20  #the number of training iterations
model      = xgb.train(param, D_train, steps)
preds      = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

#model.dump_model('test.txt')

#print("Precision = {}".format(precision_score(Y_test, best_preds, average="macro")))
#print("Recall = {}".format(recall_score(Y_test, best_preds, average="macro")))
#print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))

# Calculating different metrics
predict     = RF.predict(best_preds)  # Hva blir RF her?
accuracy 	= accuracy_score(Y_test, best_preds)
precision   = precision_score(Y_test, best_preds, average="macro")
recall      = recall_score(Y_test, best_preds, average="macro")
f1_score    = f1_score(y_test, predict)

# Calculate the absolute errors
errors      = abs(predict - y_test)

# Printing the different metrics:
func.Print_parameters(accuracy, f1_score, precision, recall, errors)


"""
RF.fit(X_train,y_train)

# Calculating different metrics
predict     = RF.predict(X_test)
accuracy 	= RF.score(X_test,y_test)
precision   = func.precision(y_test, predict)
recall      = func.recall(y_test, predict)
f1_score    = func.F1_score(y_test, predict)
"""
