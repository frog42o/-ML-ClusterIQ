import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./dataset/data.csv")

print(data.head())

data.fillna(0, inplace = True)

data["TotalPurchase"] = data["Quantity"] * data["UnitPrice"]

frequency = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
frequency.columns = ['CustomerID', 'Frequency']
data = pd.merge(data, frequency, on='CustomerID', how='left')

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
current_date = pd.Timestamp('now')
recency = data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
recency['Recency'] = (current_date - recency['InvoiceDate']).dt.days
data = pd.merge(data, recency[['CustomerID', 'Recency']], on='CustomerID', how='left')

monetary = data.groupby('CustomerID')['TotalPurchase'].sum().reset_index()
monetary.columns = ['CustomerID', 'Monetary']
data = pd.merge(data, monetary, on='CustomerID', how='left')


#RFM Segmentation
features = data[['Frequency', 'Monetary', 'Recency']]

print(features.head())

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#elbow method

wcss= []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#K = 3

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

data['Cluster'] = clusters


print(data)

plt.figure(figsize=(10,6))
sns.scatterplot(x=scaled_features[:,0], y=scaled_features[:,1], hue=data['Cluster'], palette='Set1')
plt.title('Customer Segments')
plt.show()

centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers = pd.DataFrame(centers, columns=features.columns)
print(cluster_centers)

