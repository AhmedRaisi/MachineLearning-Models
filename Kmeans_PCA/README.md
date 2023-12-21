
# Kmeans_PCA Project

## Description
The Kmeans_PCA project combines two fundamental machine learning techniques: K-means clustering and Principal Component Analysis (PCA). The project's goal is to demonstrate how K-means clustering performs on datasets both in their raw form and when transformed using PCA. PCA is employed to reduce the dimensionality of the data, potentially improving the efficiency and accuracy of the K-means clustering.

## How to Run
To run the project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Prepare the Data**: Ensure that the `Digits089.csv` file is present in the same directory as the scripts.
3. **Install Dependencies**: Install the required Python libraries, including `numpy` and `matplotlib`. You can do this using pip:
   ```
   pip install numpy matplotlib
   ```
4. **Run the Script**: Execute the main script using Python 3.
   ```
   python3 Kmeans_PCA.py
   ```

## Observations from Output
Upon running the `Kmeans_PCA.py` script, the following observations were made:

- **Performance on Raw Data**: 
  - The K-means algorithm converged in 27 iterations, taking 2.47 seconds.
  - The classification accuracy on the raw data was found to be 0.94.

- **Performance with PCA (73 Dimensions)**:
  - When the data was projected into 73 dimensions using PCA, the algorithm converged in 26 iterations, taking 1.86 seconds.
  - The classification accuracy remained at 0.94, indicating that reducing dimensions did not affect the accuracy adversely.

- **Performance with PCA (1 Dimension)**:
  - Reducing the data to 1 dimension with PCA resulted in the algorithm taking 33 iterations and 2.24 seconds to converge.
  - The classification accuracy dropped to 0.74, suggesting that while 1-dimensional representation speeds up the process, it loses significant information impacting the accuracy.

## Conclusion
The Kmeans_PCA project effectively demonstrates the impact of dimensionality reduction on the performance of K-means clustering. While PCA can enhance efficiency and maintain accuracy to a certain extent, overly aggressive dimensionality reduction can lead to loss of crucial information, as evidenced by the drop in classification accuracy in the 1-dimensional case.

