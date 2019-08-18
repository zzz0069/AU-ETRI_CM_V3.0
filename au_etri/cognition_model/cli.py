import click
import cm_baseline_v3
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, \
							 GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm

@click.command()
@click.option('--model', prompt='Choose ML model(1-Random Forest, 2-Decision Tree, 3-Gradient Boosting, 4-Logistic Regression, 5-Adaptive Boosting, 6-SVM)')

def choose_model(model):
	if model == '1':
	 	cm_baseline_v3.main(RF(n_estimators=100))
	elif model == '2':
		cm_baseline_v3.main(DT())
	elif model == '3':
		cm_baseline_v3.main(GB())
	elif model == '4':
		cm_baseline_v3.main(LR(solver='liblinear', C=1, max_iter = 500, multi_class='auto'))
	elif model == '5':
		cm_baseline_v3.main(AB())
	elif model == '6':
		cm_baseline_v3.main(svm.SVC(gamma='auto'))
	else:
		print('Please choose a integer between 1 and 6')
if __name__ == '__main__':
    choose_model()