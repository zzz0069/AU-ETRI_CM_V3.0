import os
from enum import Enum

# Root directory path "au_etri", It's just hacky :)
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

class Model(Enum):
	BEHAVIOR = "behavior"
	INFERENCE = "inference"
	COGNITION = "cognition"

class Dataset(Enum):
	TRAIN = "train"
	VALID = "valid"

class DatasetName(Enum):
	KAGGLE = "kaggle"
	AUBURN = "auburn"
	
class DatasetType(Enum):
	SMALL = "small"
	FULL = "full"

# Behavior Model
class BMBaseline(Enum):
	VGG16 = "vgg16"
	VGG16_KERAS = "vgg16_keras"
	SVM = "svm"

# Inference Model
class IMBaseline(Enum):
	MARKOV = "markov_model"
	LSTM = "lstm"
	
# Cognition Model
class CMBaseline(Enum):
	CLINGO = "clingo"
	LightGBM = "light_gbm_cm"
	CM_v3 = "cm_v3.0_baseline"

class CMModel(Enum):
	RandomForest = "random forest"
	DecisionTree = "decision tree"
	GradientBoosting = "gradient boosting"
	Logistic_Regression = "logistic regression"
	Adaptive_Boosting = "adaptive boosting"
	SVM = "svm"

	'''
	@classmethod
	def toList(self):
		return [e.value for e in self]
	'''