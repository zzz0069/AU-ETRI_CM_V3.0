
from __future__ import division
import numpy as np
import scipy
import matplotlib as mpl 
import pandas as pd
from scipy import linalg as la
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.naive_bayes import GaussianNB as NGB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn import tree, svm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from collections import Counter
import data_loaders
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import unique_labels
from cli import choose_model

class_names = ['0_back', '1_back', '2_back']

def plot_confusion_matrix_simpleified(cnf_matrix):
        
   # Show confusion matrix in a separate window
    plt.matshow(cnf_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Test")

# def ml_model(model_name):
    
#     if model_name == "logistic_regression":
#         model = LR() 
#     if model_name == "random_forest":
#         model = RF()
#     if model_name == "adaboost":
#         model = AB()
#     if model_name == "gaussian_naive_bayes":
#         model = NGB()
#     if model_name == "DT":
#         model = DT()
#     if model_name == "gradient_boosting":
#         model = GB()
#     if model_name == "svm":
#         model = svm()

#     return model

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            pass# title = 'Normalized confusion matrix'
        else:
            pass# title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['0_back', '1_back', '2_back']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass# print('Confusion matrix, without normalization')

    print(cm)

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

def main():
    x_train_data, y_train_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/train.csv")
    x_test_data, y_test_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/test.csv")

    au_can_types = [6, 9, 12]
    n_class = len(au_can_types)
    x_train = x_train_data
    y_train = pd.Categorical(y_train_data).codes
    x_test = x_test_data
    y_test = pd.Categorical(y_test_data).codes
    # clf = OneVsRestClassifier(LogisticRegression(C=1))
    clf = OneVsRestClassifier(DT())
    classifier = clf.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_predicted)
    # y_score = classifier.decision_function(x_test)
    y_score = clf.predict_proba(x_test)

    # for i in zip(y_test, y_predicted):
    # 	print(i[0],i[1],i[0]==i[1])

    print('Accuracy on training data: ',(np.sum(y_predicted == y_test)/len(y_test))*100,'%')

    # print('cnf matrix is:', cnf_matrix)
    # plot_confusion_matrix_simpleified(cnf_matrix)

    np.set_printoptions(precision=2)
     
    #Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_predicted, classes=class_names,
                          title='Confusion matrix of Decision Tree Classifier, without normalization')

    #Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_predicted, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
