import test_data
from KNN import KNN
import numpy as np

X_train, X_test, Y_train, Y_test = test_data.get_data()

knn = KNN(k=3)

knn.fit(X_train, Y_train)

y_hat = knn.predict(X_test)

performance = knn.evaluate(Y_test)

if __name__ == '__main__':

	print('Length of Y_test: {}'.format(len(Y_test)))
	print('Length of Y_hat: {}'.format(len(y_hat)))
	print('Y_hat output: {}'.format(y_hat))

	print('Model accuracy: {}%'.format(performance))