import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

merged_data = pd.read_csv('../data/interpolated_merged_data.csv')
merged_data=merged_data.fillna(0)
chosen_symptoms=['symptom:Cough', 'symptom:Common cold' , 'symptom:Fever', 'symptom:Pneumonia']

symptoms_list=[s for s in merged_data.columns.values if s.startswith('symptom:')]
regions = merged_data.groupby(merged_data['open_covid_region_code']).aggregate('count')
regions = regions[merged_data]
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
        break
#save data
merged_data.to_csv('../data/interpolated_merged_data_scaled.csv')
