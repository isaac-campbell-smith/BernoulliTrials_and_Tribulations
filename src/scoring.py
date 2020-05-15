import numpy as np 
import pandas as pd 
from sklearn import metrics

def predict(model, X, threshold=0.5):
    '''Return prediction of the fitted binary-classifier model model on X using
    the specifed `threshold`. NB: class 0 is the positive class'''
    return np.where(model.predict_proba(X)[:, 1] > threshold,
                    model.classes_[1],
                    model.classes_[0])

def confusion_matrix(model, X, y, threshold=0.5):
    cf = pd.crosstab(y, predict(model, X, threshold))
    cf = cf.rename(columns={0:'Not', 1:'Fraud'}, index={0:'Not', 1:'Fraud'})
    cf.index.name = 'actual'
    cf.columns.name = 'predicted'
    return cf

def total_cost(predicted, cb=np.array([[0,-22.5],[-27.5,-7.5]])):
    cost = predicted*cb
    return cost.values.sum()

def print_results(X, y, model, threshold=.5, p=True):
    if p:
        yhat = predict(model, X, threshold)
    else:
        yhat = model.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y, yhat).flatten()
    print ('  TP  |  FN  |  FP  |  TN  ')
    print ('--------------------------')
    print(f'{tp:6d}|{fn:6d}|{fp:6d}|{tn:6d}')
    print ('--------------------------')
    recall = metrics.recall_score(y, yhat)
    roc = metrics.roc_auc_score(y, yhat)
    precision = metrics.precision_score(y, yhat)
    print(f'Recall:     {recall:0.4f}')
    print(f'Precision:  {precision:.4f}')
    print(f'ROC_AUC:    {roc:.4f}')

