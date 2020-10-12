import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import date, timedelta

merged_data = pd.read_csv('../data/merged_data.csv')
merged_data=merged_data.fillna(0)
chosen_symptoms=['symptom:Cough', 'symptom:Common cold' , 'symptom:Fever', 'symptom:Pneumonia']

symptoms_list=[s for s in merged_data.columns.values if s.startswith('symptom:')]
#print(symptoms_list)
regions = merged_data.groupby(merged_data['open_covid_region_code']).aggregate('count')
regions = regions[merged_data]
region_names = list(regions.index)
#print(region_names)
#get region
by_state = merged_data.groupby('open_covid_region_code')

date_data = by_state.get_group(region_names[0]).sort_values(by=['date'])['date']

print(merged_data.columns[7:21])
#display symptom search per region per week
#temporarily displays one plot at a time, but will be grouped up later
for sym_index, symptom_name in enumerate(chosen_symptoms):
    #continue
    colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k'
    width = 0.8
    bottoms = np.array([0])
    tempI=0
    colIndex=0
    for index, region_name in enumerate(region_names):

        temp = by_state.get_group(region_name).sort_values(by=['date'])[symptom_name]
        #print('before:')
        #print(temp)
        temp[np.isnan(temp)] = 0

        #calculate median and divide the values in array by median
        temp = np.true_divide(temp, np.median(temp))
        #print('after:')
        #print(temp)

        #Replace all the region values in the symptom column by the median
        #for replaceI, tempEnum in enumerate(temp):
        #    merged_data.at[tempI, symptom_name] = tempEnum
        #    tempI+=1


        if colIndex >= len(colors):
            colIndex=0
        plt.bar(date_data, temp, width, bottom=bottoms, color=colors[colIndex])
        bottoms = np.add(bottoms, np.array(temp))
        colIndex+=1

    plt.title(symptom_name)
    plt.xticks(rotation=90)
    plt.legend(labels=region_names)
    plt.show()
    #break


#PCA

X=merged_data[merged_data.columns[3:len(symptoms_list)+3]]
print(X)
#print(iris.data)
# Time to get them tools --> import and initialize
from sklearn.decomposition import PCA
pca = PCA(n_components=len(symptoms_list))
pca.fit(X)
X_reduced = pca.transform(X)

# Let's make a scatter plot as before for this reduced dimension Iris data
# this formatter will label the colorbar with the correct target names
#formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

#plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap=plt.cm.get_cmap('viridis',3))
plt.scatter(X_reduced[:,0], X_reduced[:,1])
#plt.colorbar(ticks=[0,1,2], format=formatter)
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.show()


#Clusters
from sklearn.cluster import KMeans
kmeans_high = KMeans(n_clusters=3, random_state=0)
kmeans_high.fit(X)
y_pred_high = kmeans_high.predict(X)

kmeans_low = KMeans(n_clusters=3, random_state=0)
kmeans_low.fit(X_reduced)
y_pred_low = kmeans_low.predict(X_reduced)

# Plot 3 scatter plots -- two for high and low dimensional clustering results and one indicating the ground truth labels

plt.subplot(3,1,1)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_pred_high, cmap=plt.cm.get_cmap('viridis',3))
plt.colorbar(ticks=[0,1,2])
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.title("Cluster labels for high-dimensional KMeans")

plt.subplot(3,1,2)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_pred_low, cmap=plt.cm.get_cmap('viridis',3))
plt.colorbar(ticks=[0,1,2])
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.title("Cluster labels for low-dimensional KMeans")

plt.subplot(3,1,3)
#plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap=plt.cm.get_cmap('viridis',3))
plt.scatter(X_reduced[:,0], X_reduced[:,1], cmap=plt.cm.get_cmap('viridis',3))
#plt.colorbar(ticks=[0,1,2], format=formatter)
plt.clim(-0.5,2.5)
plt.xlabel("PC #1")
plt.ylabel("PC #2")
plt.title("Ground truth labels")
plt.show()