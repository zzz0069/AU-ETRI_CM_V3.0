'''
Only first eight columns (Label_ACTION and another seven features) are needed.
'''

import pandas as pd
import numpy as np
import os
import csv

FIRST_CAN_FILE = 2
LAST_CAN_FILE = 10
DATA = 48733
TRAIN_DATA = 40000
TEST_DATA = 8733
HEADERS = ['Label_ACTION', 'GTEC_TIME', 'GT_HR', 'GT_HRV', 'ECG', 'GT_EDR', 'GT_SC', 'EDA']

if not os.path.exists('../../../datasets/cognition_model/au_can_data'):
    os.makedirs('../../../datasets/cognition_model/au_can_data')

TRAIN_CSV = '../../../datasets/cognition_model/au_can_data/train.csv'
TEST_CSV = '../../../datasets/cognition_model/au_can_data/test.csv'

can_filenames = [str(i) + '.csv' for i in range(FIRST_CAN_FILE, LAST_CAN_FILE)]

# def class_csv(classname):
# 	for can_filename in can_filenames: 
# 		au_can_data = pd.read_csv('datasets/can/'+can_filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7], names=HEADERS).query('Label_ACTION == [classname]') #ZERO_BACK = 6, ONE_BACK = 9, and TWO_BACK = 12
# 		if not os.path.exists('datasets/au_can_data/'+ str(classname)):
# 			os.makedirs('datasets/au_can_data/'+ str(classname))
# 		au_can_data.to_csv('datasets/au_can_data/'+ str(classname)+'/'+str(classname)+'.csv', index=False, header=False, mode='a')

'''
main function
'''
def main():
	data = []
	if os.path.exists('../../../datasets/cognition_model/au_can_data/au_can_full_dataset/au_can_full_dataset.csv'):
		os.remove('../../../datasets/cognition_model/au_can_data/au_can_full_dataset/au_can_full_dataset.csv')
	
	'''Create related au_can_data'''
	for can_filename in can_filenames: 
		au_can_data = pd.read_csv('../../../datasets/cognition_model/can/'+can_filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7], names=HEADERS).query('Label_ACTION == [6, 9, 12]') #ZERO_BACK = 6, ONE_BACK = 9, and TWO_BACK = 12
		''' merge all csv file to one and suffle all the rows'''
		au_can_data.to_csv('../../../datasets/cognition_model/au_can_data/au_can_full_dataset.csv', index=False, header=False, mode='a')

	au_all_data = pd.read_csv('../../../datasets/cognition_model/au_can_data/au_can_full_dataset.csv')
	data = au_all_data.values.tolist()

	train_indices = np.random.choice(DATA, TRAIN_DATA, replace=False)
	residue = np.array(list(set(range(DATA)) - set(train_indices)))
	test_indices = np.random.choice(len(residue), TEST_DATA, replace=False)

	if not os.path.exists(TRAIN_CSV):
	    with open(TRAIN_CSV, "w", newline='') as train_data:
	        writer = csv.writer(train_data)
	        # writer.writerows([HEADERS])
	        writer.writerows(np.array(data)[train_indices])
	        train_data.close()

	if not os.path.exists(TEST_CSV):
	    with open(TEST_CSV, "w", newline='')as test_data:
	        writer = csv.writer(test_data)
	        # writer.writerows([HEADERS])
	        writer.writerows(np.array(data)[test_indices])
	        test_data.close()	
main()