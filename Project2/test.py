from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve

# Import project functions
import functions   as func
import credit_card as CD

features, target = CD.CreditCard()
X, y = CD.DesignMatrix(features, target)

Acc         = func.accuracy(model, y)
Acc_sklearn =
F1          = func.precision()
F1_sklearn  =
Rec         = func.recall()
Rec_sklearn =

#------------------------------------------------------------------------------
# We can test Accuracy score against scikit learn:
#------------------------------------------------------------------------------

def Accuracy():
    assert Acc == Acc_sklearn, \
    print("Our Accuracy score is not equal to the scikit learn Accuracy score.\
           Accuracy = %s, Accuracy (sklearn) = %s" %(Acc, Acc_sklearn))
    print("Our Accuracy score is equal to the scikit learn Accuracy score")

#------------------------------------------------------------------------------
# We can test F1 score against scikit learn:
#------------------------------------------------------------------------------

def F1score():
    assert F1 == F1_sklearn, \
    print("Our F1 score is not equal to the scikit learn F1 score. \
           F1 = %s, F1 (sklearn) = %s" %(F1, F1_sklearn))
    print("Our F1 score is equal to the scikit learn F1 score")

#------------------------------------------------------------------------------
# We can test Recall against scikit learn:
#------------------------------------------------------------------------------

def Recall():
    assert Rec == Rec_sklearn, \
    print("Our Recall score is not equal to the scikit learn Recall score. \
           Recall = %s, Recall (sklearn) = %s" %(Rec, Rec_sklearn))
    print("Our Recall score is equal to the scikit learn Recall score")
