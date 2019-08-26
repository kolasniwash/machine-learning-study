from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split


def get_data():

	X, y = make_blobs(n_samples=300, centers=4,
                      cluster_std=0.60, random_state=0)

	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	return X_train, X_test, Y_train, Y_test
