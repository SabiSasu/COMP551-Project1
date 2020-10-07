import pandas as pd
import numpy as np
from datetime import date, timedelta

# download and load datasets into Pandas Dataframe
hospital_data = pd.read_csv('https://raw.githubusercontent.com/google-research/open-covid-19-data/master/data/exports/cc_by/aggregated_cc_by.csv')
symptoms_data = pd.read_csv('https://raw.githubusercontent.com/google-research/open-covid-19-data/master/data/exports/search_trends_symptoms_dataset/United%20States%20of%20America/2020_US_weekly_symptoms_dataset.csv')

#
# CLEAN SYMPTOM DATA
#

num_regions = 12
region_sparseness_thresh = 0.75

symptom_names = [s for s in symptoms_data.columns.values if s.startswith('symptom:')]
num_weeks = symptoms_data['open_covid_region_code'].value_counts()[0]
symptoms_data.drop(['sub_region_2', 'sub_region_2_code'], axis=1, inplace=True)

# aggregate data by region
regions = symptoms_data.groupby(symptoms_data['open_covid_region_code']).aggregate('count')
regions = regions[symptom_names]

# drop symptoms with no data
regions.drop(columns=regions.columns[regions.sum() == 0], inplace=True)

# drop symptoms that aren't available in at least num_regions
regions.drop(columns=regions.columns[regions.astype(bool).sum(axis=0) < num_regions], inplace=True)

# drop regions with sparse or missing symptom data
regions.drop(index=regions.index[regions.sum(axis=1)/(regions.shape[1] * num_weeks) < region_sparseness_thresh], inplace=True)

# filter symptom_data based off remaining regions and symptoms
region_names = list(regions.index)
dropped_symptoms = list(set(symptom_names) - set(regions.columns.values))

filtered_symptom_data = symptoms_data.drop(dropped_symptoms, axis='columns')
filtered_symptom_data = filtered_symptom_data[symptoms_data['open_covid_region_code'].isin(region_names)]

# filtered_symptom_data.to_csv('./symptom_data.csv')
# regions.to_csv('./regions.csv')

#
# CLEAN HOSPITAL DATA
#

# drop regions with no matching symptom data
hospital_data = hospital_data[hospital_data['open_covid_region_code'].isin(region_names)]

# drop columns with no data
hospital_data.dropna(axis='columns', how='all', inplace=True)

# aggregate daily to weekly
d = dict(tuple(hospital_data.groupby('open_covid_region_code')))

aggregated_data = []

for region, region_data in d.items():

    # skip if total hospitalization data is zero
    if region_data['hospitalized_cumulative'].sum() == 0:
        continue

    # get earliest start of week date
    min_date_str = region_data['date'].min()
    min_date = date(*map(int, min_date_str.split('-')))
    start_week = min_date + timedelta(days=(7 - min_date.weekday()) % 7)
    end_week = start_week + timedelta(days=7)

    # convert date strings to actual date type
    region_data['date'] = region_data['date'].map(lambda x: date(*map(int, x.split('-'))))

    # select data from week
    weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]

    # sum hospital for each full week in the region's data set
    while weekly_data.shape[0] == 7:
        aggregated_data.append([region, start_week, weekly_data['hospitalized_cumulative'].sum(), weekly_data['hospitalized_new'].sum()])

        start_week = end_week
        end_week = start_week + timedelta(days=7)
        weekly_data = region_data[region_data['date'] >= start_week][region_data['date'] < end_week]

filtered_hospital_data = pd.DataFrame(data=aggregated_data, columns=['open_covid_region_code', 'date', 'hospitalized_cumulative', 'hospitalized_new'])

# hospital_data.to_csv('./daily_hospital_data.csv')
# filtered_hospital_data.to_csv('./weekly_hospital_data.csv')

#
# JOIN SYMPTOM AND HOSPITAL DATA
#

filtered_symptom_data['date'] = filtered_symptom_data['date'].map(lambda x: date(*map(int, x.split('-'))))
merged_data = pd.merge(filtered_symptom_data, filtered_hospital_data, on=['open_covid_region_code', 'date'])

merged_data.to_csv('../data/merged_data.csv')
