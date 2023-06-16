#pip install pandas seaborn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Correlation Matrix Demo")

# Generate random data
np.random.seed(42)
data = np.random.rand(100, 5)
columns = ["A", "B", "C", "D", "E"]
df = pd.DataFrame(data, columns=columns)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Display the correlation matrix
st.write("Correlation Matrix:")
st.dataframe(corr_matrix)

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Explanation
st.write("""
A correlation matrix is a table that shows the correlation coefficients between multiple variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses.

In this demo, we generated random data for 5 variables (A, B, C, D, and E) and calculated the correlation matrix using the Pandas library. The correlation matrix is displayed as a DataFrame and visualized using a heatmap from the Seaborn library.

The values in the correlation matrix range from -1 to 1, representing the strength and direction of the relationship between the variables. A value of 1 indicates a strong positive correlation, -1 indicates a strong negative correlation, and 0 indicates no correlation.

Keep in mind that correlation does not imply causation. A high correlation between two variables does not necessarily mean that one variable causes the other.
""")
