import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from data_loaders import readfile
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, GradientBoostingClassifier as GB
from sklearn import tree, svm
import data_loaders
def main():
        X_train_data, y_train_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/train.csv")
        X_test_data, y_test_data = data_loaders.readfile("../../datasets/cognition_model/au_can_data/test.csv")
        au_can_types = [6, 9, 12]
        n_class = len(au_can_types)
        X_train = X_train_data
        y_train = label_binarize(y_train_data, classes=[6, 9, 12])
        X_test = X_test_data
        y_test = label_binarize(y_test_data, classes=[6, 9, 12])
        n_classes = y_train.shape[1]


        # X, y = readfile('../../datasets/cognition_model/au_can_data/au_can_full_dataset.csv')
        # # Binarize the output
        # y = label_binarize(y, classes=[6, 9, 12])
        # n_classes = y.shape[1]


        # random_state = np.random.RandomState(0)

        # shuffle and split training and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
        #                                                     random_state=0)

        # Learn to predict each class against the other
        # clf = OneVsRestClassifier(svm.SVC(gamma='auto'))
        clf = OneVsRestClassifier(GB())
        classifier = clf.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        # print(y_predicted)
        print('Accuracy: ',(np.sum(y_predicted == y_test)/len(y_test)/3)*100,'%')
        y_score = classifier.decision_function(X_test)
        print('y_score_baseline: ',y_score, 'y_score shape: ', y_score.shape)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

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
        for i, color in zip(range(n_classes), colors):
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