import numpy as np

class PCA():
    def __init__(self, num_dim=None):
        """
        Initializes the PCA instance.
        
        :param num_dim: Number of dimensions to reduce to. If None, it will be determined based on variance.
        """
        self.num_dim = num_dim  # Target dimension after reduction
        self.mean = None  # Mean of the training data
        self.W = None  # Projection matrix

    def fit(self, X):
        """
        Fits the PCA model to the dataset X.

        :param X: A numpy array of shape (n_samples, n_features).
        :return: Transformed data in reduced dimension and number of dimensions.
        """
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0).reshape(1, -1)
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = eigenvalues[::-1]  # Reverse to get descending order
        eigenvectors = eigenvectors[:, ::-1]  # Reverse to match eigenvalues order

        # Determine the number of dimensions to keep
        if self.num_dim is None:
            total_variance = sum(eigenvalues)
            cumulative_variance = 0
            for i in range(len(eigenvalues)):
                cumulative_variance += eigenvalues[i]
                if cumulative_variance / total_variance > 0.9:
                    self.num_dim = i + 1  # +1 as index starts at 0
                    break

        # Determine the projection matrix
        self.W = eigenvectors[:, :self.num_dim].T

        # Project the data onto the lower-dimensional space
        X_pca = np.dot(X_centered, self.W.T)

        return X_pca, self.num_dim

    def predict(self, X):
        """
        Projects new data X onto the PCA components.

        :param X: New data, a numpy array of shape (n_samples, n_features).
        :return: Data transformed into the PCA space.
        """
        # Normalize the new data based on the training mean
        X_centered = X - self.mean

        # Project the data onto the PCA components
        X_pca = np.dot(X_centered, self.W.T)

        return X_pca

    def params(self):
        """
        Returns the parameters of the fitted PCA model.

        :return: Projection matrix, mean of the training data, and number of dimensions.
        """
        return self.W, self.mean, self.num_dim
