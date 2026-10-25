import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("data.csv")

X = data[['Hours_Studied', 'Marks']]

# Linkage matrix and dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3)
clusters = hc.fit_predict(X)

data['Cluster'] = clusters
print(data)

plt.scatter(X['Hours_Studied'], X['Marks'], c=clusters)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hierarchical Clustering")
plt.show()