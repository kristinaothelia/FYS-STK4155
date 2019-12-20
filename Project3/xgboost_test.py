"""
FYS-STK4155 - Project 3: XGBoost
"""
import numpy     as np
import xgboost   as xgb
import functions as func
import goldilock as GL

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost 				 import plot_importance
from matplotlib				 import pyplot

from sklearn.metrics         import precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score

from sklearn.metrics         import precision_score, recall_score,          \
                                    accuracy_score, confusion_matrix,       \
                                    mean_absolute_error, f1_score



# Burde brukt XGBoost classifier?
# https://www.programcreek.com/python/example/99824/xgboost.XGBClassifier?


#------------------------------------------------------------------------------
def XG_Boost(X_train, X_test, y_train, y_test, candidates, GoldiLock,   \
             feature_list, header_names, seed, Goldilock_zone=False, plot_confuse_matrix=False):



    # classifier    
    #model2 = xgb.XGBClassifier()
    
    param_test = {"max_depth": [7,8,9],
                  "n_estimator": [100,200,300,400,500,600],
                  "learning_rate": [0.1,0.3]
                  }
    
    gsearch = GridSearchCV(xgb.XGBClassifier(), param_grid = param_test, cv=5)
    model2 = gsearch.fit(X_train, y_train)
    
    #train_model1 = model1.fit(X_train, y_train)
    #train_model2 = model2.fit(X_train, y_train)
    
    #model2.fit(X_train, y_train)
    #model2.predict(X_test)
    
    #pred1 = train_model1.predict(X_test)
    #pred2 = train_model2.predict(X_test)
    
    #print("Accuracy for model 1: %g" % accuracy_score(y_test, pred1))
    #print("Accuracy for model 2: %g" % accuracy_score(y_test, pred2))
    
    #best_preds = pred2    
    #model = model2
    best_preds = model2.predict(X_test)

    #model.dump_model('test.txt')

    # Calculating different metrics
    accuracy 	= accuracy_score(y_test, best_preds)
    precision   = precision_score(y_test, best_preds, average="macro")
    recall      = recall_score(y_test, best_preds, average="macro")
    F1_score    = f1_score(y_test, best_preds, average="macro") # SE PAA DENNE!!! Feil input..
    # Calculate the absolute errors
    errors      = abs(best_preds - y_test) # SE PAA DENNE!!! Blir veldig hoy verdi. Feil input..

    # Printing the different metri
    func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='XGBoost')

    # Confusion matrix?


    # Hmmm... resultatet blir veldig rart... 90% eksoplaneter

    #D_test     = xgb.DMatrix(candidates, label=candidates)
    D_test = candidates    
    #y_pred     = model.predict(D_test)
    y_pred = model2.predict(D_test) 
    
    #pred_cand  = np.asarray([np.argmax(line) for line in y_pred])
    pred_cand = y_pred
    
    #print(pred_cand)

    # Divide into predicted false positives and confirmed exoplanets
    pred_FP    = (pred_cand == 0).sum() 	# Predicted false positives
    pred_Conf  = (pred_cand == 1).sum() 	# Predicted exoplanets/confirmed

    # Information print to terminal
    print('\nThe XGBoost method predicted')
    print('--------------------------------------')
    print('%-5g exoplanets      of %g candidates'  %(pred_Conf, len(pred_cand)))
    print('%-5g false positives of %g candidates'  %(pred_FP, len(pred_cand)))

    #plot_importance(y_pred)
    #pyplot.show()


    
    # Usikker om dette blir riktig... Predikter alle til exoplanets, ingen false
    """
    if Goldilock_zone:

        print("Goldilock zone calculations")

        D_test             = xgb.DMatrix(GoldiLock, label=GoldiLock)
        y_pred             = model.predict(D_test)
        predict_goldilocks = np.asarray([np.argmax(line) for line in y_pred])
        np.save('GoldiLock_predicted', predict_goldilocks)

        predicted_false_positive_goldilocs  = (predict_goldilocks == 0).sum()
        predicted_exoplanets_goldilocks     = (predict_goldilocks == 1).sum()

        # Information print to terminal
        print('\nThe XGBoost method predicted')
        print('--------------------------------------')
        print('%-3g exoplanets      of %g candidates'  %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
        print('%-3g false positives of %g candidates'  %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

        # Plotting a bar plot of candidates predicted as confirmed and false positives
        # Need to fix input title, labels etc maybe?
    
        func.Histogram2(predict_goldilocks)

        GL.GoldilocksZone()
    
    """
