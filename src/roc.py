import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from catboost import Pool


def plot_roc(X, y, clf, plot_name, ax, cat=False, **kwargs):
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    n_splits=5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        if cat:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = y.iloc[train_index]
            feats = clf.get_params()['cat_features']
            cat_data = Pool(X_train, y_train, cat_features=feats)
            clf.fit(cat_data)
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            #clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y.iloc[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(plot_name + ' Receiver operating characteristic')
    ax.legend(loc="lower right")
