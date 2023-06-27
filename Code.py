import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Read data from CSV
df = pd.read_csv("data.csv")
X = df["X"].values.reshape(-1, 1)
y = df["y"].values

# Perform Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred = poly_model.predict(X_poly)

# Sort data for plotting
sort_indices = np.argsort(X.squeeze())
X_sorted = X[sort_indices]
y_pred_sorted = y_pred[sort_indices]

# Visualize the results
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X_sorted, y_pred_sorted, color="red", label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression")
plt.show()
