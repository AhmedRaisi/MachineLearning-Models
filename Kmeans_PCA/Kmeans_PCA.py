import numpy as np
from matplotlib import pyplot as plt
from Mykmeans import Kmeans
from MyPCA import PCA
import time

# Function to plot the error history
def plot_error_history(error_history, title, filename):
    fig = plt.figure()
    plt.plot(np.arange(len(error_history)), error_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    fig.set_size_inches(10, 10)
    fig.savefig(filename)
    plt.close(fig)

# Read in data
data = np.genfromtxt("Digits089.csv", delimiter=",")
Xtrain = data[data[:,0] != 5, 2:]
ytrain = data[data[:,0] != 5, 1]
Xtest = data[data[:,0] == 5, 2:]
ytest = data[data[:,0] == 5, 1]

# Apply kmeans algorithm to raw data
clf = Kmeans(k=8)
start = time.time()
num_iter, error_history = clf.fit(Xtrain, ytrain)
time_raw = time.time() - start

# Plot the history of reconstruction error
plot_error_history(error_history, 'Error History for Raw Data', 'raw_data.png')

# Using kmeans clustering for classification
predicted_label = clf.predict(Xtest)
acc_raw = np.count_nonzero(predicted_label == ytest) / len(Xtest)

# Apply kmeans algorithm to low-dimensional data (PCA) that captures >90% of variance
pca = PCA()
Xtrain_pca, num_dim = pca.fit(Xtrain)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca, error_history_pca = clf.fit(Xtrain_pca, ytrain)
time_pca = time.time() - start

# Plot the history of reconstruction error for PCA
plot_error_history(error_history_pca, 'Error History for PCA Data', 'pca.png')

# Using kmeans clustering for classification on PCA data
Xtest_pca = pca.predict(Xtest)
predicted_label = clf.predict(Xtest_pca)
acc_pca = np.count_nonzero(predicted_label == ytest) / len(Xtest)

# Apply kmeans algorithm to 1D feature obtained from PCA
pca = PCA(num_dim=1)
Xtrain_pca, _ = pca.fit(Xtrain)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca_1, error_history_pca_1 = clf.fit(Xtrain_pca, ytrain)
time_pca_1 = time.time() - start

# Plot the history of reconstruction error for 1D PCA
plot_error_history(error_history_pca_1, 'Error History for 1D PCA Data', 'pca_1d.png')

# Using kmeans clustering for classification on 1D PCA data
Xtest_pca = pca.predict(Xtest)
predicted_label = clf.predict(Xtest_pca)
acc_pca_1 = np.count_nonzero(predicted_label == ytest) / len(Xtest)

# Print information
print('Using raw data converged in %d iteration (%.2f seconds)' % (num_iter, time_raw))
print('Classification accuracy: %.2f' % acc_raw)

print('#################')
print('Project data into %d dimensions with PCA converged in %d iteration (%.2f seconds)' % (num_dim, num_iter_pca, time_pca))
print('Classification accuracy: %.2f' % acc_pca)

print('#################')
print('Project data into 1 dimension with PCA converged in %d iteration (%.2f seconds)' % (num_iter_pca_1, time_pca_1))
print('Classification accuracy: %.2f' % acc_pca_1)
