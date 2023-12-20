
# Gaussian Discriminant Analysis Models

## Overview
This directory contains the implementation of Gaussian Discriminant Analysis (GDA) models in Python. GDA is a probabilistic classification model used in machine learning for classifying data points into predefined classes. It assumes that the data from each class follows a Gaussian (normal) distribution.

## Files
- `GaussianModel.py`: The main Python script that contains the implementation of GDA models. This includes two types of GDA:
  - Standard GDA with class-dependent covariance.
  - GDA with class-independent (shared) covariance.
  - GDA with diagonal covariance matrix, assuming feature independence.

## Running the Model
To execute the model and see the results, run the following command in the terminal:
```
python3 GaussianModel.py
```

## Output Explanation
Upon running the script, you will see confusion matrices for each type of GDA model. Here's an example output:

```
Confusion Matrix for Gaussian Discriminant with class-dependent covariance
[[14  5]
 [16 65]]

Confusion Matrix for Gaussian Discriminant with class-independent covariance
[[24  5]
 [ 6 65]]

Confusion Matrix for Gaussian Discriminant with diagonal covariance
[[26 20]
 [ 4 50]]
```

Each confusion matrix represents the performance of the respective GDA model on classifying test data. The matrices are in the format:

```
[[True Positives, False Positives]
 [False Negatives, True Negatives]]
```

## Analysis of Results
1. **Class-Dependent Covariance:**
   - True Positives (TP): 14
   - False Positives (FP): 5
   - False Negatives (FN): 16
   - True Negatives (TN): 65
   - Observations: This model shows a moderate balance between identifying positives and negatives, with some misclassifications.

2. **Class-Independent Covariance:**
   - TP: 24
   - FP: 5
   - FN: 6
   - TN: 65
   - Observations: This model performs better in identifying positive cases compared to the class-dependent model, suggesting that assuming a shared covariance might be reasonable for this dataset.

3. **Diagonal Covariance:**
   - TP: 26
   - FP: 20
   - FN: 4
   - TN: 50
   - Observations: While this model has the highest number of TPs, it also has a significantly higher number of FPs. This indicates a trade-off where the model is more sensitive to positives but at the cost of accuracy, possibly due to the assumption of feature independence.

## Conclusion
The performance of each GDA model varies based on the assumptions made about the covariance matrix. Choosing the right model depends on the specific characteristics and requirements of the dataset and the problem at hand.

