import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from datetime import date, timedelta

#TASK 3: SUPERVISED LEARNING
#PART 1

#get the new dataset and load it into pandas dataframe
merged_data = pd.read_csv("../data/merged_data.csv")

#split into training and validation sets based on region (80% train 20% validate)
regions = merged_data.groupby(merged_data['open_covid_region_code']).aggregate('count')
region_names = list(regions.index) #list of all the region names
slice = (int) (len(region_names) / 5)

region_groups = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

for i in range(5):
    for j in range(slice):
        region_groups[i] = region_groups[i].append(merged_data[merged_data['open_covid_region_code'] == region_names[i * slice + j]])

if (slice * 5 < len(region_names)): #number of regions doesn't divide equally in 5 parts
    region_groups[0] = region_groups[0].append(merged_data[merged_data['open_covid_region_code'] == region_names[5 * slice]])
if (slice * 5 + 1 < len(region_names)):
    region_groups[1] = region_groups[1].append(merged_data[merged_data['open_covid_region_code'] == region_names[5 * slice + 1]])
if (slice * 5 + 2 < len(region_names)):
    region_groups[2] = region_groups[2].append(merged_data[merged_data['open_covid_region_code'] == region_names[5 * slice + 2]])
if (slice * 5 + 3 < len(region_names)):
    region_groups[3] = region_groups[3].append(merged_data[merged_data['open_covid_region_code'] == region_names[5 * slice + 3]])


#split into training and validation sets based on time (data after a timestamp for validate, before for train)
dates = merged_data.groupby(merged_data['date']).aggregate('count')
dates_span = list(dates.index) #list of all dates
slice = dates_span.index("2020-08-10") + 1 #keep dates after 2020-08-10

time_training = pd.DataFrame()
time_validation = pd.DataFrame()

for a in range(slice):
    time_training = time_training.append(merged_data[merged_data['date'] == dates_span[a]])

for b in range(len(dates_span) - slice):
    time_validation = time_validation.append(merged_data[merged_data['date'] == dates_span[a + 1 + b]])
    #print(time_validation.iloc[:,6])

#PART 2

#predict hospitalization based on symptom search

#KNN regression performance with regions (5-fold cross validation)
region_training = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()] #5-fold cross validation
region_validation = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

for c in range(5):
    region_validation[c] = region_groups[c]
    for d in range(5):
        if (d != c):
            region_training[c] = region_training[c].append(region_groups[d])

symptom_names = [s for s in merged_data.columns.values if s.startswith('symptom:')]

start = merged_data.columns.get_loc(symptom_names[0])
end = merged_data.columns.get_loc(symptom_names[-1])

#assuming all symptom columns are one next to another in merged_data

k = [1, 5, 15]

for m in range(len(k)): #trying different K's
    for n in range(5):
        X_train = region_training[n].iloc[:, start:(end + 1)].values #symptoms
        y_train = region_training[n].iloc[:, -1].values #new hospitalizations
        X_valid = region_validation[n].iloc[:, start:(end + 1)].values
        y_valid = region_validation[n].iloc[:, -1].values
        print(region_training[n])
        print("NEXT")
        print(X_train)

        regressor = KNeighborsRegressor(n_neighbors=k[m])
        regressor.fit(X_train, y_train)
        y_predicted = regressor.predict(X_valid)
        print(y_predicted)
#should we perform scaling?

#KNN regression performance with date

#decision tree regression performance with regions

#decision tree regression performance with date

 