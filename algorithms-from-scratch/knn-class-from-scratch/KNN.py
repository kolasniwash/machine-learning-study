

import numpy as np
from collections import Counter

class KNN:

	def __init__(self, k = 3):
		self.k = k
		self.Y_hat = []

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		self.X_test = X_test

		#returns all predictions for each row in X_test

		for t in range(X_test.shape[0]):
			prediction = self.make_prediction(self.X_train, X_test[t,:])	

			self.Y_hat.append(prediction)

		return self.Y_hat


	def make_prediction(self, X_train, predict):

		#makes one prediction!
		distance = []

		for i in range(X_train.shape[0]):

			euclidian_distance = self.euclidean_distances(X_train[i,:], predict)

			distance.append(euclidian_distance)

		distance_classifier = list(zip(np.array(distance), self.y_train))

		#sort the array and select the k lowest pair
		nearest_neighbours = sorted(distance_classifier)[:self.k]

		#get teh categorical values of the k nearest and put in list
		votes = [vote[1] for vote in nearest_neighbours]
		
		#count the highest votes and update to y_hat
		Y_val = (Counter(votes).most_common(1)[0][0])

		return Y_val

	def euclidean_distances(self, feature, predict):
		return np.linalg.norm(feature - predict, axis=0)

	def evaluate(self, Y_test):
		self.Y_test = Y_test

		correct = np.sum(np.where(Y_test == self.Y_hat, 1, 0))
		total = len(Y_test)

		return (correct/total)*100