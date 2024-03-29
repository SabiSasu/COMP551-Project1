import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler


#https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models

#PCA
merged_data = pd.read_csv('../data/interpolated_merged_data_scaled.csv')
symptoms_list=[s for s in merged_data.columns.values if s.startswith('symptom:')]
X=merged_data[merged_data.columns[4:len(symptoms_list)+4]]
print(X)

#Figure out how many components (kinda useless)
pca2 = PCA()
pca2.fit_transform(StandardScaler().fit_transform(X))
num_pc_components = len(pca2.explained_variance_ratio_)
plt.tight_layout()
plt.subplot(2,1,1)
plt.plot(np.linspace(1,num_pc_components,num_pc_components),100*pca2.explained_variance_ratio_)
plt.xlabel("Principal Component #")
plt.ylabel("% Variance explained")
plt.title("Variance Explained vs Principal Components")

plt.subplot(2,1,2)
plt.plot(np.linspace(1,num_pc_components,num_pc_components),100*np.cumsum(pca2.explained_variance_ratio_))
plt.plot(np.linspace(1,num_pc_components,num_pc_components),95*np.ones((num_pc_components,)),'k--')
plt.xlabel("Principal Component #")
plt.ylabel("% Variance explained")
plt.title("Cumulative Variance Explained vs Principal Components")
plt.tight_layout()
plt.show()


pca = PCA(n_components=20)
principalComponents = pca.fit_transform(StandardScaler().fit_transform(X))# Plot the explained variances
#principalComponents = pca.fit(X)# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.show()

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

#PCA graph
markersize=4
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
plt.scatter(X_reduced[:,0], X_reduced[:,1], s=markersize)
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.title("Fit and transformed only data")
plt.show()

#PCA graph with standardized data
markersize=4
pca = PCA(n_components=4)
X_reduced = pca.fit_transform(StandardScaler().fit_transform(X))
plt.scatter(X_reduced[:,0], X_reduced[:,1], s=markersize)
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.title("Standardized and scaled data")
plt.show()


#knee rule to determine number of clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(PCA_components.iloc[:, :3])
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

ks = range(2, 6)
for k in ks:
    #Clusters
    clusterNum=k
    #high is raw data
    kmeans_high = KMeans(n_clusters=clusterNum, random_state=0)
    kmeans_high.fit(X)
    y_pred_high = kmeans_high.predict(X)

    #low is reduced dimensionality data
    kmeans_low = KMeans(n_clusters=clusterNum, random_state=0)
    kmeans_low.fit(X_reduced)
    y_pred_low = kmeans_low.predict(X_reduced)

    # Plot 3 scatter plots -- two for high and low dimensional clustering results and one indicating the ground truth labels

    plt.subplot(2,1,1)
    plt.scatter(X_reduced[:,0], X_reduced[:,1], s=markersize, c=y_pred_high)
    plt.colorbar(ticks=[0,1,2])
    plt.clim(-0.5,2.5)
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.title("Cluster labels for high-dimensional KMeans")

    plt.subplot(2,1,2)
    plt.scatter(X_reduced[:,0], X_reduced[:,1], s=markersize, c=y_pred_low)
    plt.colorbar(ticks=[0,1,2])
    plt.clim(-0.5,2.5)
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.title("Cluster labels for low-dimensional KMeans")

    plt.tight_layout()
    plt.show()