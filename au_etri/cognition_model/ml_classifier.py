from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import linalg as la, interp
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB,\
                             GradientBoostingClassifier as GB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

import data_loaders
from cli import choose_model
class_names = ['0_back', '1_back', '2_back']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['0_back', '1_back', '2_back']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass# print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def main(ml_model):
    X_train_data, y_train_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/train.csv")
    X_test_data, y_test_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/test.csv")

    au_can_types = [6, 9, 12]
    n_class = len(au_can_types)
    X_train = X_train_data
    y_train = pd.Categorical(y_train_data).codes
    X_test = X_test_data
    y_test = pd.Categorical(y_test_data).codes
    clf = OneVsRestClassifier(ml_model)
    # clf = ml_model
    classifier = clf.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predicted)
    print('Accuracy: ',(np.sum(y_predicted == y_test)/len(y_test))*100,'%')
    np.set_printoptions(precision=2)
     
    #Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_predicted, classes=class_names,
                          title='Confusion matrix without normalization')

    #Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_predicted, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    y_score = classifier.decision_function(X_test)
    y_test = label_binarize(y_test_data, classes=[6, 9, 12])
    # print('y_score_classifier: ',y_score, 'y_score shape: ', y_score.shape)
    # if ml_model == svm.SVC(gamma='auto'):
    #     y_score = classifier.decision_function(X_test)
    # else:
    #     y_score = clf.predict_proba(X_test)

       # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0}_back (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()