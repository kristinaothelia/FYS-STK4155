"""
run: pytest -W ignore::DeprecationWarning
"""

# Import project functions
import functions   as func
import credit_card as CD

features, target = CD.CreditCard()
X, y       = CD.DesignMatrix(features, target)

# Splitting X and y in a train and test set
X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=0)

eta = 0.01
gamma = 0.1  # learning rate?

# Calculating the beta values based og the training set
betas_train = func.steepest(X_train, y_train, gamma)
#betas_train = func.SGD_beta(X_train, y_train, eta, gamma)

# Calculating ytilde and the model of logistic regression
z         = X_test @ betas_train   # choosing best beta here?
model     = func.logistic_function(z)
model     = func.IndicatorFunc(model)




acc_scikit, TPR_scikit, precision_scikit, f1_score_scikit, AUC_scikit, predict_proba_scikit = func.scikit(X_train, X_test, y_train, y_test, model)

Acc         = func.accuracy(model, y_test)
Acc_sklearn = acc_scikit
F1          = func.F1_score(y_test, model)
F1_sklearn  = f1_score_scikit
Rec         = func.recall(y_test, model)
Rec_sklearn = TPR_scikit
#precision          = func.precision(y_test, model)

#------------------------------------------------------------------------------
# We can test Accuracy score against scikit learn:
#------------------------------------------------------------------------------

def test_Accuracy():
    assert Acc == Acc_sklearn, \
    print("Our Accuracy score is not equal to the scikit learn Accuracy score.\
           Accuracy = %s, Accuracy (sklearn) = %s" %(Acc, Acc_sklearn))
    print("Our Accuracy score is equal to the scikit learn Accuracy score")

#------------------------------------------------------------------------------
# We can test F1 score against scikit learn:
#------------------------------------------------------------------------------

def test_F1score():
    assert F1 == F1_sklearn, \
    print("Our F1 score is not equal to the scikit learn F1 score. \
           F1 = %s, F1 (sklearn) = %s" %(F1, F1_sklearn))
    print("Our F1 score is equal to the scikit learn F1 score")

#------------------------------------------------------------------------------
# We can test Recall against scikit learn:
#------------------------------------------------------------------------------

def test_Recall():
    assert Rec == Rec_sklearn, \
    print("Our Recall score is not equal to the scikit learn Recall score. \
           Recall = %s, Recall (sklearn) = %s" %(Rec, Rec_sklearn))
    print("Our Recall score is equal to the scikit learn Recall score")
