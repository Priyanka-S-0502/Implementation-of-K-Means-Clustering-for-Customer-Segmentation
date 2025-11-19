# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Import data, inspect it, and handle missing values if any.
2.Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.
3.Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.
4.Assign cluster labels to each data point.
5.Plot data points in a scatter plot, color-coded by cluster assignments for interpretation.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: PRIYANKA S
RegisterNumber: 212224040255  
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Basic info
print("NAME : PRIYANKA S")
print("REG NO:212224040255")
print(data.head())
print(data.info())

# Implementation-of-KMeans-Clustering-for-Customer-Segmentation/README.md at main · Deepikaasuresh304/Implementation-of-KMeans-Clustering-for-Customer-Segmentation/blob/main/README.md
print(data.isnull().sum())

# Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# Apply KMeans with 5 clusters
km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:])

# Add cluster label to the data
data["cluster"] = y_pred

# Separate the clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Visualize the clusters
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:

<img width="1072" height="593" alt="image" src="https://github.com/user-attachments/assets/df78bd7c-3cf7-48ff-b10b-806ed050d060" />

<img width="1287" height="188" alt="image" src="https://github.com/user-attachments/assets/2b3dd434-a9b7-4f39-b338-abb467fb2618" />

<img width="990" height="545" alt="image" src="https://github.com/user-attachments/assets/4d309383-4437-4625-b91a-49cbdd45cd64" />

<img width="837" height="533" alt="image" src="https://github.com/user-attachments/assets/07290e05-73c5-4215-9598-49c8364cde77" />

<img width="887" height="540" alt="image" src="https://github.com/user-attachments/assets/2a8d62e5-0eb1-4da8-8b9d-f52f07a5b237" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
