import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from datetime import date, timedelta
from scipy import stats

#TASK 3: SUPERVISED LEARNING
#PART 1

#get the new dataset and load it into pandas dataframe
merged_data = pd.read_csv("../data/interpolated_merged_data.csv")
merged_data = merged_data.loc[np.abs(stats.zscore(merged_data['hospitalized_new'])) < 3]

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

k = [1, 10, 20, 25, 30, 33, 35, 37, 40, 43, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300]

region_k_array_results = np.empty((len(k), 3)) #k, mean of costs for 5-fold, r2

for m in range(len(k)): #trying different K's
    five_fold_cost = [0, 0, 0, 0, 0]
    five_fold_r2 = [0, 0, 0, 0, 0]

    for n in range(5):
        X_train = region_training[n].iloc[:, start:(end + 1)].values #symptoms
        y_train = region_training[n].iloc[:, -1].values #new hospitalizations
        X_valid = region_validation[n].iloc[:, start:(end + 1)].values
        y_valid = region_validation[n].iloc[:, -1].values
        #print(region_training[n])
        #print("NEXT")
        #print(X_train)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)

        if k[m] > len(X_train):
            k[m] = len(X_train)
            
        regressor = KNeighborsRegressor(n_neighbors=k[m])
        regressor.fit(X_train, y_train)
        y_predicted = regressor.predict(X_valid)
        #print(X_valid.shape)
        #print(y_predicted.shape)
        #print(X_valid)
        #print(n)
        #print(y_predicted)
        #print("")
        #print("real y")
        #print("Y VALID")
        #print(y_valid)
        #print("")
        cost = 0
        
        for u in (range(len(y_predicted))):
            error = np.abs(y_predicted[u] - y_valid[u])
            #print(error)
            cost = cost + error

        cost = (cost/(len(y_predicted))) #using MAE
        #print(cost)
        five_fold_cost[n] = cost
        five_fold_r2[n] = r2_score(y_valid, y_predicted)
    
    #calculate the mean and variance for 5-fold
    mean = (five_fold_cost[0] + five_fold_cost[1] + five_fold_cost[2] + five_fold_cost[3] + five_fold_cost[4])/5
    mean_r2 = (five_fold_r2[0] + five_fold_r2[1] + five_fold_r2[2] + five_fold_r2[3] + five_fold_r2[4])/5
    
    region_k_array_results[m][0] = k[m]
    region_k_array_results[m][1] = mean
    region_k_array_results[m][2] = mean_r2

#the best KNN with regions
region_best_mean = float("inf")
region_best_index = 0
#print( [i[1] for i in region_k_array_results])
region_standard_dev_of_Ks = np.std([i[1] for i in region_k_array_results])

for t in range(len(region_k_array_results)):
    if (region_k_array_results[t][1] < region_best_mean):
        region_best_mean = region_k_array_results[t][1]
        region_best_index = t
#print(region_k_array_results)
#print(region_standard_dev_of_Ks)
#print(region_k_array_results[region_best_index])
#print(region_best_mean)
#print(region_standard_dev_of_Ks)

acceptable_model = region_best_mean + region_standard_dev_of_Ks
region_acceptable_mean = 0
region_acceptable_index = 0
#loop again to find the best model within 1 standard deviation of the model with lowest mean
for t in range(len(region_k_array_results)):
    if (region_k_array_results[t][1] < acceptable_model):
        region_acceptable_mean = region_k_array_results[t][1]
        region_acceptable_index = t
        break

#KNN regression performance with date
time_k_array_results = np.empty((len(k), 3)) #k, cost, r2

for m in range(len(k)): #trying different K's
    
    X_train = time_training.iloc[:, start:(end + 1)].values #symptoms
    y_train = time_training.iloc[:, -1].values #new hospitalizations
    X_valid = time_validation.iloc[:, start:(end + 1)].values
    y_valid = time_validation.iloc[:, -1].values
    #print(region_training[n])
    #print("NEXT")
    #print(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    if k[m] > len(X_train):
            k[m] = len(X_train)

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
        error = np.abs(y_predicted[u] - y_valid[u])
        #print(error)
        cost = cost + error

    cost = (cost/(len(y_predicted)))
    #print(cost)
        
    time_k_array_results[m][0] = k[m]
    time_k_array_results[m][1] = cost
    time_k_array_results[m][2] = r2_score(y_valid, y_predicted)

#the best KNN with dates
time_best_cost = float("inf")
time_best_index = 0
time_standard_dev_of_Ks = np.std([i[1] for i in time_k_array_results])

for t in range(len(time_k_array_results)):
    if (time_k_array_results[t][1] < time_best_cost):
        time_best_cost = time_k_array_results[t][1]
        time_best_index = t

#print(time_k_array_results)
#print(time_best_index)

acceptable_model = time_best_cost + time_standard_dev_of_Ks
time_acceptable_cost = 0
time_acceptable_index = 0

#loop again to find the best model within 1 standard deviation of the model with lowest mean
for t in range(len(time_k_array_results)):
    if (time_k_array_results[t][1] < acceptable_model):
        time_acceptable_cost = time_k_array_results[t][1]
        time_acceptable_index = t
        break

#decision tree regression performance with regions

five_fold_cost = [0, 0, 0, 0, 0]
five_fold_r2 = [0, 0, 0, 0, 0]

for n in range(5):
     X_train = region_training[n].iloc[:, start:(end + 1)].values #symptoms
     y_train = region_training[n].iloc[:, -1].values #new hospitalizations
     X_valid = region_validation[n].iloc[:, start:(end + 1)].values
     y_valid = region_validation[n].iloc[:, -1].values

     regressor = DecisionTreeRegressor(random_state=0)
     regressor.fit(X_train, y_train)
     y_predicted = regressor.predict(X_valid)

     cost = 0
     for u in (range(len(y_predicted))):
        error = np.abs(y_predicted[u] - y_valid[u])
        #print(error)
        cost = cost + error

     cost = (cost/(len(y_predicted)))
     #print(cost)
     five_fold_cost[n] = cost
     five_fold_r2[n] = r2_score(y_valid, y_predicted)

region_decision_tree_avg_cost = (five_fold_cost[0] + five_fold_cost[1] + five_fold_cost[2] + five_fold_cost[3] + five_fold_cost[4])/5 
region_decision_tree_avg_r2 = (five_fold_r2[0] + five_fold_r2[1] + five_fold_r2[2] + five_fold_r2[3] + five_fold_r2[4])/5  

#print(region_decision_tree_avg_cost)


#decision tree regression performance with date

X_train = time_training.iloc[:, start:(end + 1)].values #symptoms
y_train = time_training.iloc[:, -1].values #new hospitalizations
X_valid = time_validation.iloc[:, start:(end + 1)].values
y_valid = time_validation.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_valid)

cost = 0
for u in (range(len(y_predicted))):
    error = np.abs(y_predicted[u] - y_valid[u])
    #print(error)
    cost = cost + error

cost = (cost/(len(y_predicted)))
r2 = r2_score(y_valid, y_predicted)
#print(region_k_array_results)
#print(time_k_array_results)

print("KNN with regions average MAE cost: " + str(region_best_mean) + " with optimal k = " + str(region_k_array_results[region_best_index][0]) + " and optimal r2 score = " + str(region_k_array_results[region_best_index][2]))
print("But choosing the simplest model within 1 standard deviation " + str(region_standard_dev_of_Ks) + " of the best model: " + str(region_acceptable_mean) + " with acceptable k = " + str(region_k_array_results[region_acceptable_index][0]) + " and acceptable r2 score = " + str(region_k_array_results[region_acceptable_index][2]))
print("KNN with dates MAE cost: " + str(time_best_cost) + " with optimal k = " + str(time_k_array_results[time_best_index][0]) + " and optimal r2 score = " + str(time_k_array_results[time_best_index][2]))
print("But choosing the simplest model within 1 standard deviation " + str(time_standard_dev_of_Ks) + " of the best model: " + str(time_acceptable_cost) + " with acceptable k = " + str(time_k_array_results[time_acceptable_index][0]) + " and acceptable r2 score = " + str(time_k_array_results[time_acceptable_index][2]))
print("decision tree with region average MAE cost: " + str(region_decision_tree_avg_cost) + " and r2 score: " + str(region_decision_tree_avg_r2))
print("decision tree with dates MAE cost: " + str(cost) + " and r2 score: " + str(r2))

#linear regression performance with regions

five_fold_cost = [0, 0, 0, 0, 0]
five_fold_r2 = [0, 0, 0, 0, 0]

for n in range(5):
     X_train = region_training[n].iloc[:, start:(end + 1)].values #symptoms
     y_train = region_training[n].iloc[:, -1].values #new hospitalizations
     X_valid = region_validation[n].iloc[:, start:(end + 1)].values
     y_valid = region_validation[n].iloc[:, -1].values

     regressor = linear_model.LinearRegression()
     regressor.fit(X_train, y_train)
     y_predicted = regressor.predict(X_valid)

     cost = 0
     for u in (range(len(y_predicted))):
        error = np.abs(y_predicted[u] - y_valid[u])
        #print(error)
        cost = cost + error

     cost = (cost/(len(y_predicted)))
     #print(cost)
     five_fold_cost[n] = cost
     five_fold_r2[n] = r2_score(y_valid, y_predicted)

region_linear_reg_avg_cost = (five_fold_cost[0] + five_fold_cost[1] + five_fold_cost[2] + five_fold_cost[3] + five_fold_cost[4])/5  
region_linear_reg_avg_r2 = (five_fold_r2[0] + five_fold_r2[1] + five_fold_r2[2] + five_fold_r2[3] + five_fold_r2[4])/5  
#print(region_linear_reg_avg_cost)

#linear regression performance with date

X_train = time_training.iloc[:, start:(end + 1)].values #symptoms
y_train = time_training.iloc[:, -1].values #new hospitalizations
X_valid = time_validation.iloc[:, start:(end + 1)].values
y_valid = time_validation.iloc[:, -1].values

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_valid)

cost = 0
for u in (range(len(y_predicted))):
    error = np.abs(y_predicted[u] - y_valid[u])
    #print(error)
    cost = cost + error

cost = (cost/(len(y_predicted)))
r2 = r2_score(y_valid, y_predicted)

print("linear regression with region average MAE cost: " + str(region_linear_reg_avg_cost) + " and r2 score: " + str(region_linear_reg_avg_r2))
print("linear regression with dates MAE cost: " + str(cost) + " and r2 score: " + str(r2))