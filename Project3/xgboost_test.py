"""
FYS-STK4155 - Project 3: XGBoost
"""
import numpy     as np
import xgboost   as xgb
import functions as func

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
<<<<<<< HEAD
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error

X = np.load("features.npy", allow_pickle=True)
y = np.load("targets.npy", allow_pickle=True)

y = y.astype('int')
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

print("Shape of training data:", X_train.shape)
print("Shape of testing data: ", X_test.shape)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)

param = {
    "eta": 0.3,
    "max_depth": 3,
    "objective": "multi:softprob",
    "num_class": 2}
steps = 20  #the number of training iterations

model = xgb.train(param, D_train, steps)
preds = model.predict(D_test)
=======
from sklearn.metrics         import precision_score, recall_score,          \
                                    accuracy_score, mean_squared_error,     \
                                    mean_absolute_error, f1_score
#------------------------------------------------------------------------------
seed = 0

X    = np.load("features.npy", allow_pickle=True)
y    = np.load("targets.npy", allow_pickle=True)
y    = y.astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

D_train    = xgb.DMatrix(X_train, label=Y_train)
D_test     = xgb.DMatrix(X_test, label=Y_test)

param      = {  "eta": 0.3,
                "max_depth": 3,
                "objective": "multi:softprob",
                "num_class": 2  }
steps      = 20  #the number of training iterations
model      = xgb.train(param, D_train, steps)
preds      = model.predict(D_test)
>>>>>>> ca3cf982f5e73b4d3126fe1738a1c6ae78e97775
best_preds = np.asarray([np.argmax(line) for line in preds])

#model.dump_model('test.txt')

<<<<<<< HEAD
print(50*"-")
print("Precision = {}".format(precision_score(Y_test, best_preds, average="macro")))
print("Recall = {}".format(recall_score(Y_test, best_preds, average="macro")))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
=======
>>>>>>> ca3cf982f5e73b4d3126fe1738a1c6ae78e97775

# Calculating different metrics
accuracy 	= accuracy_score(Y_test, best_preds)
precision   = precision_score(Y_test, best_preds, average="macro")
recall      = recall_score(Y_test, best_preds, average="macro")
f1_score    = f1_score(Y_test, best_preds, average="macro") # SE PAA DENNE!!! Feil input..
# Calculate the absolute errors
errors      = abs(best_preds - Y_test) # SE PAA DENNE!!! Blir veldig hoy verdi. Feil input..

# Printing the different metrics
func.Print_parameters(accuracy, f1_score, precision, recall, errors, name='XGBoost')
