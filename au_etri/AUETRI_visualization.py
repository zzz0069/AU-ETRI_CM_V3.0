import os
import numpy as np
import pandas as pd

import keras

import platform
if platform.system() == 'Darwin':
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sn

class AUETRIVisualization(keras.callbacks.Callback):

	def __init__(self):
		pass

# Accuracy plot
class AUETRIVizAccuracy(AUETRIVisualization):

	def __init__(self, path):
		self.path = path

	def on_train_begin(self, logs={}):
		self.accuracy = {'batch':[], 'epoch':[]}
		self.val_acc = {'batch':[], 'epoch':[]}
		self.logs = {'batch':[], 'epoch':[]}

	def on_batch_end(self, batch, logs={}):
		self.logs['batch'].append(logs)
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_acc['batch'].append(logs.get('val_acc'))
		self.update_plot(name="batch_accuracy.png", acc=self.accuracy['batch'], val_acc=self.val_acc['batch'])

	def on_epoch_end(self, epoch, logs={}):
		self.logs['epoch'].append(logs)
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_acc['epoch'].append(logs.get('val_acc'))
		self.update_plot(name="epoch_accuracy.png", acc=self.accuracy['epoch'], val_acc=self.val_acc['epoch'])

	def update_plot(self, name="test_plot.png", **args):
		N = np.arange(0, len(args.get("acc")))
		plt.figure()
		plt.plot(N, args.get("acc"), label = "train_acc")
		plt.plot(N, args.get("val_acc"), label = "val_acc")
		plt.title("Accuracy Plot")
		plt.grid(True)
		plt.xlabel("Iteration #")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, name))
		plt.close()

# Loss plot
class AUETRIVizLoss(AUETRIVisualization):

	def __init__(self, path):
		self.path = path

	def on_train_begin(self, logs={}):
		self.loss = {'batch':[], 'epoch':[]}
		self.val_loss = {'batch':[], 'epoch':[]}
		self.logs = {'batch':[], 'epoch':[]}

	def on_batch_end(self, batch, logs={}):
		self.logs['batch'].append(logs)
		self.loss['batch'].append(logs.get('loss'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.update_plot(name="batch_loss.png", loss=self.loss['batch'], val_loss=self.val_loss['batch'])

	def on_epoch_end(self, epoch, logs={}):
		self.logs['epoch'].append(logs)
		self.loss['epoch'].append(logs.get('loss'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.update_plot(name="epoch_loss.png", loss=self.loss['epoch'], val_loss=self.val_loss['epoch'])

	def update_plot(self, name="test_plot.png", **args):
		N = np.arange(0, len(args.get("loss")))
		plt.figure()
		plt.plot(N, args.get("loss"), label = "train_loss")
		plt.plot(N, args.get("val_loss"), label = "val_loss")
		plt.title("Loss Plot")
		plt.grid(True)
		plt.xlabel("Iteration #")
		plt.ylabel("Loss")
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, name))
		plt.close()

# SVM plot
class AUETRIVizSVM():

	def __init__(self, path):
		self.path = path

	def plot(self, total_num_current, num_true_class_list):
		plt.figure()
		plt.plot(total_num_current, num_true_class_list)
		plt.title("SVM accuracy plot")
		plt.grid(True)
		plt.xlabel("Test images #")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, "svm_accuracy.png"))
		plt.close()

# LSTM plot
class AUETRIVizInference():

	def __init__(self, path):
		self.path = path

	def plot_cost(self, epoch_list, perplexity_list):
		plt.figure()
		plt.title("LSTM cost plot")
		plt.plot(epoch_list, perplexity_list)
		plt.xlabel('epoch')
		plt.ylabel('perpelxity')
		plt.grid(True)
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, "lstm_cost.png"))
		plt.close()

	def plot_probability_matrix(self, prob, classes=[]):
		df_cm = pd.DataFrame(prob, index=classes, columns=classes)
		plt.figure(figsize=(16,12))
		plt.title("LSTM probability matrix")
		sn.heatmap(df_cm, annot=True)
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, "lstm_prob_matrix.png"))
		plt.close()


# MARKOV Model plot
class AUETRIVizMarkov():

	def __init__(self, path):
		self.path = path

	def plot_cost(self, epoch_list, perplexity_list):
		plt.figure()
		plt.title("LSTM cost plot")
		plt.plot(epoch_list, perplexity_list)
		plt.xlabel('epoch')
		plt.ylabel('perpelxity')
		plt.grid(True)
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, "lstm_cost.png"))
		plt.close()

	def plot_probability_matrix(self, prob, classes=[]):
		df_cm = pd.DataFrame(prob, index=classes, columns=classes)
		plt.figure(figsize=(16,12))
		plt.title("LSTM probability matrix")
		sn.heatmap(df_cm, annot=True)
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.path, "lstm_prob_matrix.png"))
		plt.close()