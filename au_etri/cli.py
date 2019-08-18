# -*- coding: utf-8 -*-

"""Console script for au_etri."""
import sys
import os
from enum import Enum
from AUETRI_constants import Model, DatasetType, DatasetName, BMBaseline, IMBaseline, CMBaseline, CMModel
from AUETRI_constants import PROJECT_ROOT_DIR
from AUETRI_file_manager import AUETRIFileManager
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, \
                             GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm

import cognition_model.cm_baseline_v3 as cm_baseline_v3

def createShowableList(input_enum):
    outputList = []
    
    enumToList = list(input_enum)
    for i in range(len(enumToList)):
        text = str(i+1) + '-' + enumToList[i].value
        outputList.append(text)
    listToStr = '[' + ' '.join(outputList) + ']' + ':'

    return enumToList, outputList, listToStr

def CM_ML_model(model):
    if model == '1':
        cm_baseline_v3.main(RF(n_estimators=100))
    elif model == '2':
        cm_baseline_v3.main(DT())
    elif model == '3':
        cm_baseline_v3.main(GB())
    elif model == '4':
        cm_baseline_v3.main(LR(solver='lbfgs', C=1, max_iter=1000, multi_class='auto'))
    elif model == '5':
        cm_baseline_v3.main(AB())
    elif model == '6':
        cm_baseline_v3.main(svm.SVC(gamma='auto'))

def main():
    # model
    modelEnumToList, modelList, listToStr = createShowableList(Model)
    modelMsg = 'choose_model' + listToStr
    while True:
        modelUserInput = input(modelMsg)
        # <TODO: possible bug if you are hacker>
        if int(modelUserInput) > len(modelList):
            print('Entered input is not a valid number.')
        else:
            break

    # baseline
    if modelUserInput == '1':
        baselineEnumToList, baselineList, listToStr = createShowableList(BMBaseline)
    elif modelUserInput == '2':
        baselineEnumToList, baselineList, listToStr = createShowableList(IMBaseline)
    elif modelUserInput == '3':
        baselineEnumToList, baselineList, listToStr = createShowableList(CMBaseline)
    baselineMsg = 'choose_baseline' + listToStr   
    while True:
        baselineUserInput = input(baselineMsg)
        if int(baselineUserInput) > len(baselineList):
            print('Entered input is not a valid number.')
        else:
            break

    #ml model for CM
    CMMLModelEnumToList, CMMLModelList, listToStr = createShowableList(CMModel)
    if baselineUserInput == '3':
        print('au can dataset has been selected.')   
        while True:
            CMMLModelMsg = 'choose_CM_ML_model' + listToStr
            CMMLModelUserInput = input(CMMLModelMsg)
            CM_ML_model(CMMLModelUserInput)
            if int(CMMLModelUserInput) > len(CMMLModelList):
                print('Entered input is not a valid number.')
            else:
                break           

    # dataset name
    datasetNameEnumToList, datasetNameList, listToStr = createShowableList(DatasetName)
    datasetNameMsg = 'choose_dataset_name' + listToStr
    datasetNameUserInput = 0
    if int(modelUserInput) == 1:
        while True:
            datasetNameUserInput = input(datasetNameMsg)
            if int(datasetNameUserInput) > len(datasetNameList):
                print('Entered input is not a valid number.')
            else:
                break

    # dataset
    datasetEnumToList, datasetList, listToStr = createShowableList(DatasetType)
    datasetMsg = 'choose_dataset' + listToStr
    while True:
        if int(modelUserInput) == 3 and int(baselineUserInput) == 3:
            datasetUserInput = 0
            break
        else:
            datasetUserInput = input(datasetMsg)
        # <TODO: possible bug if you are hacker>
        if int(datasetUserInput) > len(datasetList):
            print('Entered input is not a valid number.')
        else:
            break

    model = modelEnumToList[int(modelUserInput) - 1]
    baseline = baselineEnumToList[int(baselineUserInput) - 1].value
    datasetName = datasetNameEnumToList[int(datasetNameUserInput) - 1].value
    datasetType = datasetEnumToList[int(datasetUserInput) - 1].value
    filepath = AUETRIFileManager.get_baseline_file_path(model, baseline)
    runFile = 'python ' + os.path.join(PROJECT_ROOT_DIR, 'au_etri', filepath) + ' ' + '--dataset_name=' + datasetName + ' ' + '--dataset_type=' + datasetType 
    os.system(runFile)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover