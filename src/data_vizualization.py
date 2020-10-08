import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import date, timedelta

merged_data = pd.read_csv('./merged_data.csv')

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




#Clusters