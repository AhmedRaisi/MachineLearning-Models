import numpy as np

class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        # Initialize the Gaussian Discriminant model
        self.mean = np.zeros((k, d))  # Initialize mean vector for each class
        self.shared_cov = shared_cov  # Flag to use shared covariance matrix or not
        if self.shared_cov:
            self.S = np.zeros((d, d))  # Shared covariance matrix for all classes
        else:
            self.S = np.zeros((k, d, d))  # Separate covariance matrix for each class
        if priors is not None:
            self.p = priors  # Prior probabilities of each class
        else:
            self.p = [1.0 / k for i in range(k)]  # Assume equal priors if not provided
        self.k = k  # Number of classes
        self.d = d  # Number of features

    def fit(self, Xtrain, ytrain):
        # Fit the model to training data
        first_class = Xtrain[ytrain == 1]  # Data points belonging to the first class
        second_class = Xtrain[ytrain == 2]  # Data points belonging to the second class
        first_class_mean = np.mean(first_class, axis=0)  # Mean of the first class
        second_class_mean = np.mean(second_class, axis=0)  # Mean of the second class

        # Set the mean for each class
        for i in range(2):
            if i == 0:
                self.mean[i, :] = first_class_mean
            elif i == 1:
                self.mean[i, :] = second_class_mean

        # Compute covariance matrix
        if self.shared_cov:
            # Compute shared covariance matrix for class-independent covariance
            combined_data = np.vstack((first_class, second_class))
            self.S = np.cov(combined_data.T, ddof=0)
        else:
            # Compute separate covariance matrices for class-dependent covariance
            self.S[0] = np.cov(first_class.T, ddof=0)
            self.S[1] = np.cov(second_class.T, ddof=0)

    def predict(self, Xtest):
        # Predict class labels for the test set
        predicted_class = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            best_class = None
            best_discriminant_value = -np.inf
            for c in range(self.k):
                mean = self.mean[c]
                cov = self.S if self.shared_cov else self.S[c]
                # Compute the discriminant function value
                inv_cov = np.linalg.inv(cov)
                term1 = np.dot((Xtest[i] - mean).T, inv_cov)
                term2 = np.dot(term1, (Xtest[i] - mean))
                discriminant_value = -0.5 * term2 - 0.5 * np.log(np.linalg.det(cov)) + np.log(self.p[c])
                if discriminant_value > best_discriminant_value:
                    best_discriminant_value = discriminant_value
                    best_class = c + 1
            predicted_class[i] = best_class
        return predicted_class

    def params(self):
        # Return the learned parameters (means and covariance matrices)
        if self.shared_cov:
            return self.mean, self.S
        else:
            return self.mean, self.S


class GaussianDiscriminant_Diagonal(GaussianDiscriminant):
    def fit(self, Xtrain, ytrain):
        # Fit the model with diagonal covariance matrix
        super().fit(Xtrain, ytrain)
        # Modify covariance matrices to be diagonal (assuming feature independence)
        if not self.shared_cov:
            for i in range(self.k):
                self.S[i] = np.diag(np.diag(self.S[i]))

    def predict(self, Xtest):
        # Predict class labels with diagonal covariance assumption
        predicted_class = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            best_class = None
            best_discriminant_value = -np.inf
            for c in range(self.k):
                mean = self.mean[c]
                var = self.S if self.shared_cov else self.S[c]
                # Handle zero variance
                var = np.where(var == 0, 1e-10, var)
                inv_var = 1. / var
                term1 = (Xtest[i] - mean) ** 2
                term2 = term1 * inv_var
                discriminant_value = -0.5 * np.sum(term2) - 0.5 * np.sum(np.log(var)) + np.log(self.p[c])
                if discriminant_value > best_discriminant_value:
                    best_discriminant_value = discriminant_value
                    best_class = c + 1
            predicted_class[i] = best_class
        return predicted_class
