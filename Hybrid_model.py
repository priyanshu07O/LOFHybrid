import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert time to seconds since midnight
df['Time'] = pd.to_datetime(df['Time']).dt.hour * 3600 + pd.to_datetime(df['Time']).dt.minute * 60 + pd.to_datetime(df['Time']).dt.second

# Scale the numerical features
scaler = StandardScaler()
df[['Temperature', 'Humidity', 'Time']] = scaler.fit_transform(df[['Temperature', 'Humidity', 'Time']])

# Extract selected attributes
X = df[['Time', 'Temperature', 'Humidity']]

# Fit LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['LOF_Score'] = lof.fit_predict(X)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
df['DBSCAN_Labels'] = dbscan.fit_predict(X)

# Plot LOF scores
plt.figure(figsize=(7, 5))
plt.scatter(df['Temperature'], df['Humidity'], c=df['LOF_Score'], cmap='viridis', s=50, alpha=0.5)
plt.colorbar(label='LOF Score')
plt.title('Local Outlier Factor (LOF)')
plt.xlabel('Temperature (Scaled)')
plt.ylabel('time(scaled)')
plt.show()

# Plot DBSCAN clusters
plt.figure(figsize=(7, 5))
plt.scatter(df['Temperature'], df['Humidity'], c=df['DBSCAN_Labels'], cmap='viridis', s=50, alpha=0.5)
plt.colorbar(label='Cluster Label')
plt.title('DBSCAN Clusters')
plt.xlabel('Temperature (Scaled)')
plt.ylabel('time(scaled)')
plt.show()


df['Hybrid_Score'] = df['LOF_Score'] * (1 - df['DBSCAN_Labels'])

hybrid_outliers = df[df['Hybrid_Score'] < 0]
plt.scatter(hybrid_outliers['Temperature'], hybrid_outliers['Humidity'], color='green', label='Hybrid Outliers')

# Plot settings
plt.xlabel('Temperature (Scaled)')
plt.ylabel('Time (Scaled)')
plt.title('Outliers Detected hybrid Model')
plt.legend()
plt.show()
