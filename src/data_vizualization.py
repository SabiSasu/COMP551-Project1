import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import date, timedelta

merged_data = pd.read_csv('../data/merged_data.csv')

#chosen_symptons=['Angular cheilitis', 'Aphonia' , 'Burning Chest Pain', 'Crackles', 'Dysautonomia', 'Hemolysis','Laryngitis','Myoclonus','Polydipsia','Rectal pain','Shallow breathing','Urinary urgency','Ventricular fibrillation','Viral pneumonia']

symptoms_list=[s for s in merged_data.columns.values if s.startswith('symptom:')]
#print(symptoms_list)
regions = merged_data.groupby(merged_data['sub_region_1']).aggregate('count')
regions = regions[merged_data]
region_names = list(regions.index)
#print(region_names)
#get region
by_state = merged_data.groupby('sub_region_1')

date_data = by_state.get_group(region_names[0]).sort_values(by=['date'])['date']
symptom_data = by_state.get_group(region_names[0]).sort_values(by=['date'])['symptom:Angular cheilitis']

data_maine_angular = {'Date': by_state.get_group(region_names[0]).sort_values(by=['date'])['date'],
        'symptom_data': by_state.get_group(region_names[0]).sort_values(by=['date'])['symptom:Angular cheilitis']}

symptom_median = symptom_data[0]
ratio=[]
ratio.append(symptom_data[0])

#might not need ratio calculation, but in case we do i'll leave what i started here
for index, data in enumerate(data_maine_angular['symptom_data'], 1):
    #calculate ratio
    ratio.append(round(data/symptom_median, 1))
    #recalculate median
    symptom_median = ((symptom_median*index)+data)/(index+1)

data_maine_angular['ratio']=ratio
#print(data_maine_angular)


#display symptom search per region per week
#temporarily displays one plot at a time, but will be grouped up later
for sym_index, symptom_name in enumerate(symptoms_list):
    continue
    colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    width = 0.8
    bottoms = np.array([0])

    for index, region_name in enumerate(region_names):
        if index == 4:
            continue

        #drop extra weeks if there are any
        temp = by_state.get_group(region_name)
        temp = temp.sort_values(by=['date'])[symptom_name]
        temp[np.isnan(temp)] = 0

        #calculate median and divide the values in array by median
        temp = np.true_divide(temp, np.median(temp))


        plt.bar(date_data, temp, width, bottom=bottoms, color=colors[index])
        bottoms = np.add(bottoms, np.array(temp))
        print(bottoms)

    plt.title(symptom_name)
    plt.xticks(rotation=90)
    plt.legend(labels=region_names)
    plt.show()
    #breaks for now until I decide on most popular symptoms
    break


#PCA
from sklearn import datasets
iris = datasets.load_iris()

#X,y = [[0.6 0.4] [1.2 0.5] [2.1 0.2]], symptoms_list
#X=iris.data
merged_data=merged_data.fillna(0)
X=merged_data[merged_data.columns[7:21]]
print(X)
print(iris.data)
# Time to get them tools --> import and initialize
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
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
# Import the KMeans module from sklearn
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