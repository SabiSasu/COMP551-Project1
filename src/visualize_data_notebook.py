#!/usr/bin/env python
# coding: utf-8

# # Task 2: Visualize and Cluster the Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from ipywidgets import widgets, interactive


# In[2]:


# Get merged data
merged_data = pd.read_csv('../data/unscaled_merged_data.csv')

symptom_names = [s for s in merged_data.columns.values if s.startswith('symptom:')]
region_names = merged_data['open_covid_region_code'].unique()


# ## Popularity of Symptoms

# In[3]:


hospitalization = widgets.Dropdown(
    options=[True, False],
    value=False,
    description='Show hospitalizations: ',
)

symptom = widgets.Dropdown(
    options=symptom_names,
    value='symptom:Fever',
    description='Symptoms: ',
)

region = widgets.Dropdown(
    options=['All of USA'] + list(region_names),
    value='All of USA',
    description='Regions: ',
)

def plot_graph(region, symptom, hospitalization):
    regions = region_names
    alpha = 0.2

    fig, symptom_plt = plt.subplots()
    hosp_plt = symptom_plt.twinx()

    # Aggregate hospital data across regions
    hospital_data = merged_data.groupby('date').sum().filter(['hospitalized_new'])

    # Adjust values if selecting individual region
    if region != 'All of USA':
        regions = [region]
        alpha = 0.9
        hospital_data = merged_data.loc[merged_data['open_covid_region_code'] == region].groupby('date').sum().filter(['hospitalized_new'])

    # Show hospitalization data only if selected
    if hospitalization:
        hosp_plt.set_ylabel('New Hospitalizations', color='tab:blue')
        hosp_plt.tick_params(axis='y', labelcolor='tab:blue')
        hosp_plt.plot(hospital_data.index.values, hospital_data['hospitalized_new'].values, color='tab:blue', alpha=0.8)

    for index, region_name in enumerate(regions):

        plot_data = merged_data.groupby('open_covid_region_code').get_group(region_name).sort_values(by=['date'])[['date', symptom]]

        date_data = plot_data['date']
        temp = plot_data[symptom]
        temp = np.true_divide(temp, np.median(temp))

        symptom_plt.plot(date_data, temp, color='tab:red', alpha=alpha)

    symptom_plt.tick_params(axis='y', labelcolor='tab:red')
    symptom_plt.set_ylabel('Relative Search Frequency', color='tab:red')
    symptom_plt.set_xlabel('Date')
    fig.tight_layout()
    fig.autofmt_xdate(rotation=80)
    plt.title(symptom[len('symptom:'):] + " : " + region)
    plt.show()

interactive(plot_graph, region=region, symptom=symptom, hospitalization=hospitalization)


# In[4]:


chosen_symptoms=['symptom:Cough', 'symptom:Common cold' , 'symptom:Fever']

symptoms_list = [s for s in merged_data.columns.values if s.startswith('symptom:')]
regions = merged_data.groupby(merged_data['open_covid_region_code']).aggregate('count')
region_names = list(regions.index)

by_state = merged_data.groupby('open_covid_region_code')

date_data = by_state.get_group(region_names[0]).sort_values(by=['date'])['date']

for sym_index, symptom_name in enumerate(symptoms_list):
    colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k'
    width = 0.8
    bottoms = np.array([0])
    tempI=0
    colIndex=0
    for index, region_name in enumerate(region_names):

        temp = by_state.get_group(region_name).sort_values(by=['date'])[symptom_name]
        temp[np.isnan(temp)] = 0

        #calculate median and divide the values in array by median
        temp = np.true_divide(temp, np.median(temp))

        #Replace all the region values in the symptom column by the median
        for replaceI, tempEnum in enumerate(temp):
            merged_data.at[tempI, symptom_name] = tempEnum
            tempI+=1


        if colIndex >= len(colors):
            colIndex=0
        if symptom_name in chosen_symptoms:
            plt.bar(date_data, temp, width, bottom=bottoms, color=colors[colIndex])
            bottoms = np.add(bottoms, np.array(temp))
            colIndex+=1

    if symptom_name in chosen_symptoms:
        plt.title(symptom_name)
        plt.xticks(rotation=90)
        plt.legend(labels=region_names, title='states', loc='upper left',bbox_to_anchor=(1.05, 1), ncol=2)
        plt.plot(figsize=(20,10))
        plt.show()


# ## PCA Function

# In[5]:


def run_pca(symptom_list, clusters):

    #PCA
    X=merged_data[symptom_list]

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

    n_components = min(20, len(symptom_list))
    pca = PCA(n_components=n_components)
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

    ks = clusters
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


# ## Principal Component Analysis (on all symptoms)

# In[6]:


all_symptoms = [s for s in merged_data.columns.values if s.startswith('symptom:')]

run_pca(all_symptoms, range(2, 6))


# ## Principal Component Analysis (on common Covid-19 symptoms)

# In[7]:


covid_symptoms = covid_symptoms=['symptom:Common cold', 'symptom:Cough', 'symptom:Fever', 'symptom:Infection']

run_pca(covid_symptoms, [2])


# In[ ]:




