#pip install scikit-learn pandas matplotlib

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression Demo")

# Generate random data for linear regression
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot the results
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color="blue", label="Actual")
ax.scatter(X_test, y_pred, color="red", label="Predicted")
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")

st.pyplot(fig)

