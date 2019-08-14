import click
import ml_classifier
import ml_baseline
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, \
							 GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm

@click.command()
@click.option('--model', prompt='Choose ML model(1-Random Forest, 2-Decision Tree, 3-Gradient Boosting, 4-Logistic Regression, 5-Adaptive Boosting, 6-SVM)')

def choose_model(model):
	if model == '1':
	 	ml_classifier.main(RF(n_estimators=100))
	elif model == '2':
		ml_classifier.main(DT())
	elif model == '3':
		ml_classifier.main(GB())
	elif model == '4':
		ml_classifier.main(LR(solver='lbfgs', C=1, multi_class='auto'))
	elif model == '5':
		ml_classifier.main(AB())
	elif model == '6':
		ml_classifier.main(svm.SVC(gamma='auto'))

if __name__ == '__main__':
    choose_model()