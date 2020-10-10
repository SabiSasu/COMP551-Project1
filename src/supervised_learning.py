import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from datetime import date, timedelta

#TASK 3: SUPERVISED LEARNING
#PART 1

#get the new dataset and load it into pandas dataframe
merged_data = pd.read_csv("../data/interpolated_merged_data.csv")

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
#SHOULD WE PERFORM SCALING?

k = [1, 3, 5, 10, 15, 20]

k_array_results = np.empty((len(k), 3))

for m in range(len(k)): #trying different K's
    five_fold_cost = [0, 0, 0, 0, 0]

    for n in range(5):
        X_train = region_training[n].iloc[:, start:(end + 1)].values #symptoms
        y_train = region_training[n].iloc[:, -1].values #new hospitalizations
        X_valid = region_validation[n].iloc[:, start:(end + 1)].values
        y_valid = region_validation[n].iloc[:, -1].values
        #print(region_training[n])
        #print("NEXT")
        #print(X_train)

        regressor = KNeighborsRegressor(n_neighbors=k[m])
        regressor.fit(X_train, y_train)
        y_predicted = regressor.predict(X_valid)
        #print(X_valid.shape)
        #print(y_predicted.shape)
        #print(X_valid)
        #print(y_predicted)
        #print("Y VALID")
        #print(y_valid)
        cost = 0
        
        for u in (range(len(y_predicted))):
            error = (y_predicted[u] - y_valid[u])**2
            #print(error)
            cost = cost + error

        cost = cost/(len(y_predicted))
        #print(cost)
        five_fold_cost[n] = cost
    
    #calculate the mean and variance for 5-fold
    mean = (five_fold_cost[0] + five_fold_cost[1] + five_fold_cost[2] + five_fold_cost[3] + five_fold_cost[4])/5
    variance = np.var(five_fold_cost)
    
    k_array_results[m][0] = k[m]
    k_array_results[m][1] = mean
    k_array_results[m][2] = variance

#print(k_array_results) 
best_mean = float("inf")
best_index = 0

for t in range(len(k_array_results)):
    if (k_array_results[t][1] < best_mean):
        best_mean = k_array_results[t][1]
        best_index = t
#IN THE NOTES IT SAYS TO CHOOSE THE SIMPLEST MODEL WITHIN 1 STANDARD DEVIATION BUT IDK WHAT HE MEANS BY THT
print(best_index)


#KNN regression performance with date

#decision tree regression performance with regions

#decision tree regression performance with date

 