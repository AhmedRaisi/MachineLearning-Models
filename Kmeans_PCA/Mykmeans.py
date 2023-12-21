import numpy as np

class Kmeans:
    def __init__(self, k=8):
        """
        Initialize the Kmeans instance.

        :param k: Number of clusters.
        """
        self.num_cluster = k  # Number of clusters
        self.center = None  # Cluster centers
        self.cluster_label = np.zeros([k,])  # Class labels for clusters
        self.error_history = []  # History of error for each iteration

    def fit(self, X, y):
        """
        Fit the K-means model to the data.

        :param X: Data points, numpy array of shape (n_samples, n_features).
        :param y: True labels, numpy array of shape (n_samples,).
        :return: Number of iterations to converge and error history.
        """
        # Predefined indices for initial cluster centers
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]

        # Initialize variables for iteration
        num_iter = 0  # Iteration count
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        # Initialize cluster centers based on predefined indices
        self.center = np.zeros((self.num_cluster, X.shape[1]))
        for i in range(len(init_idx)):
            self.center[i] = X[init_idx[i]]

        while not is_converged:
            # Assign each data point to the nearest cluster
            for i in range(len(X)):
                closest_dist = np.inf
                for c in range(self.num_cluster):
                    dist = np.linalg.norm(X[i] - self.center[c])
                    if dist < closest_dist:
                        closest_dist = dist
                        cluster_assignment[i] = c

            # Update cluster centers
            for c in range(self.num_cluster):
                assigned_points = X[cluster_assignment == c]
                if len(assigned_points) > 0:
                    self.center[c] = np.mean(assigned_points, axis=0)

            # Compute error and check for convergence
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)
            is_converged = np.array_equal(cluster_assignment, prev_cluster_assignment)
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # Assign labels to clusters based on majority voting
        for c in range(self.num_cluster):
            votes = np.bincount(y[cluster_assignment == c].astype('int'))
            self.cluster_label[c] = np.argmax(votes) if len(votes) > 0 else -1

        return num_iter, self.error_history

    def predict(self, X):
        """
        Predict the cluster labels for new data points.

        :param X: New data points, numpy array of shape (n_samples, n_features).
        :return: Predicted cluster labels for the data points.
        """
        predictions = np.zeros([len(X),])
        for i in range(len(X)):
            # Find the nearest cluster center
            closest = np.argmin([np.linalg.norm(X[i] - center) for center in self.center])
            predictions[i] = self.cluster_label[closest]
        return predictions

    def compute_error(self, X, cluster_assignment):
        """
        Compute the reconstruction error.

        :param X: Data points.
        :param cluster_assignment: Cluster assignments for each data point.
        :return: Total reconstruction error.
        """
        error = sum(np.linalg.norm(X[i] - self.center[cluster_assignment[i]])**2 for i in range(len(X)))
        return error

    def params(self):
        """
        Get the parameters of the model.

        :return: Cluster centers and cluster labels.
        """
        return self.center, self.cluster_label
