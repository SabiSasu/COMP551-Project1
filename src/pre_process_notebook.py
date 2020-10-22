#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import date, timedelta


# # Task 1: Preprocess Data

# In[2]:


# download and load datasets into Pandas Dataframe
hospital_data = pd.read_csv('https://raw.githubusercontent.com/google-research/open-covid-19-data/master/data/exports/cc_by/aggregated_cc_by.csv')
daily_symptom_data = pd.read_csv('https://raw.githubusercontent.com/google-research/open-covid-19-data/master/data/exports/search_trends_symptoms_dataset/United%20States%20of%20America/2020_US_daily_symptoms_dataset.csv')

# drop unnecessary columns
daily_symptom_data.drop(['country_region', 'country_region_code', 'sub_region_1', 'sub_region_1_code', 'sub_region_2', 'sub_region_2_code'], axis=1, inplace=True)


# ## Clean symptom data

# In[3]:


# Allow symptom data to have some missing data points
region_sparseness_thresh = 0.9

symptom_names = [s for s in daily_symptom_data.columns.values if s.startswith('symptom:')]
num_weeks = daily_symptom_data['open_covid_region_code'].value_counts()[0]

# Split by regions
daily_symptom_data.dropna(axis='columns', how='all', inplace=True)

region_symptoms = dict(tuple(daily_symptom_data.groupby('open_covid_region_code')))

# Drop symptoms from regions that are incomplete
for region, region_data in region_symptoms.items():
    region_data.dropna(axis='columns', how='all', inplace=True)
    region_data = region_data.loc[: , (region_data.count() >= num_weeks*region_sparseness_thresh)]
    region_symptoms[region] = region_data.loc[: , (region_data.count() >= num_weeks*region_sparseness_thresh)]

filtered_symptom_data = pd.concat(region_symptoms, ignore_index=True)
filtered_symptom_data.dropna(axis='columns', how='any', inplace=True)

region_symptoms = dict(tuple(filtered_symptom_data.groupby('open_covid_region_code')))


# ## Interpolate symptom data

# In[4]:


interpolated_region_symptoms = dict()
for region, region_data in region_symptoms.items():
    for s in region_data.columns.values:
        if s.startswith('symptom:'):
            region_data[s] = region_data[s].interpolate()

    interpolated_region_symptoms[region] = region_data

interpolated_symptom_data = pd.concat(interpolated_region_symptoms, ignore_index=True)


# ## Normalize symptom data

# In[5]:


normalized_region_symptoms = dict()
for region, region_data in region_symptoms.items():
    symptom_medians = []

    # calculate symptom median
    for s in region_data.columns.values:
        if s.startswith('symptom:'):
            symptom_medians.append(np.nanmedian(region_data[s]))

    # calculate region median
    region_median = np.median(symptom_medians)

    # normalize by region median
    for s in region_data.columns.values:
        if s.startswith('symptom:'):
            region_data[s] = region_data[s].div(region_median)

    normalized_region_symptoms[region] = region_data

normalized_symptom_data = pd.concat(normalized_region_symptoms, ignore_index=True)


# ## Convert daily symptom data to weekly symptom data

# In[6]:


# group symptom data by region

def daily_to_weekly(df):

    d = dict(tuple(df.groupby('open_covid_region_code')))

    summed_daily_data = []
    symptoms = [s for s in df.columns.values if s.startswith('symptom:')]

    for region, region_data in d.items():

        # get earliest start of week date
        min_date_str = region_data['date'].min()
        min_date = date(*map(int, min_date_str.split('-')))
        start_week = min_date + timedelta(days=(7 - min_date.weekday()) % 7)
        end_week = start_week + timedelta(days=7)

        # convert date strings to actual date type
        region_data['date'] = region_data['date'].map(lambda x: date(*map(int, x.split('-'))))

        # select data from week
        weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]
        weekly_data = weekly_data[symptoms]

        # sum daily symptom data for each full week in the region's data set
        while weekly_data.shape[0] == 7:
            col_vals = [start_week, region]
            col_vals.extend(weekly_data.mean(skipna=False).values)
            summed_daily_data.append(col_vals)

            start_week = end_week
            end_week = start_week + timedelta(days=7)
            weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]
            weekly_data = weekly_data[symptoms]

    return pd.DataFrame(data=summed_daily_data, columns=df.columns.values)

symptoms_data = daily_to_weekly(normalized_symptom_data)
unscaled_symptoms_data = daily_to_weekly(interpolated_symptom_data)


# ## Clean hospital data

# In[7]:


# drop non USA regions
hospital_data = hospital_data[hospital_data['open_covid_region_code'].str.contains('US-')]

# drop irrelavent data
hospital_data = hospital_data.filter(['open_covid_region_code', 'date', 'hospitalized_new'])

# aggregate daily to weekly
d = dict(tuple(hospital_data.groupby('open_covid_region_code')))

aggregated_data = []
min_start_date = date(*map(int, hospital_data['date'].min().split('-')))

for region, region_data in d.items():

    # skip if total hospitalization data is zero
    if region_data['hospitalized_new'].sum() == 0:
        continue

    # get earliest start of week date
    min_date_str = region_data['date'].min()
    min_date = date(*map(int, min_date_str.split('-')))
    start_week = min_date + timedelta(days=(7 - min_date.weekday()) % 7)
    end_week = start_week + timedelta(days=7)

    # Set common start date
    min_start_date = max(min_start_date, start_week)

    # convert date strings to actual date type
    region_data['date'] = region_data['date'].map(lambda x: date(*map(int, x.split('-'))))

    # select data from week
    weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]

    # sum hospital for each full week in the region's data set
    while weekly_data.shape[0] == 7:
        aggregated_data.append([region, start_week, weekly_data['hospitalized_new'].sum()])

        start_week = end_week
        end_week = start_week + timedelta(days=7)
        weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]

filtered_hospital_data = pd.DataFrame(data=aggregated_data, columns=['open_covid_region_code', 'date', 'hospitalized_new'])

# Drop all rows before common start date
filtered_hospital_data = filtered_hospital_data[filtered_hospital_data['date'] >= min_start_date]


# ## Join symptom and hospital data

# In[8]:


unscaled_symptoms_data.to_csv('../data/temp.csv')
merged_data = pd.merge(symptoms_data, filtered_hospital_data, on=['open_covid_region_code', 'date'])
unscaled_merged_data = pd.merge(unscaled_symptoms_data, filtered_hospital_data, on=['open_covid_region_code', 'date'])

# Remove outliers for supervised learning
from scipy import stats
merged_data = merged_data.loc[np.abs(stats.zscore(merged_data['hospitalized_new'])) < 3]

merged_data.to_csv('../data/merged_data.csv')
unscaled_merged_data.to_csv('../data/unscaled_merged_data.csv')


# In[ ]:




