# -*- coding: utf-8 -*-
"""
Tushar Iyer
"""
import pandas as pd
import math
import datetime
from matplotlib import pyplot as plt
import numpy as np
from itertools import compress, starmap
from operator import gt
from operator import lt
    
#----------INSTANCE VARIABLES----------
ts_ep = []
ts_est = []
ts_date = []
ts_day = []
i = 0

#----------IMPORT CSV INTO DATAFRAME AND SAVE ts_second AS LIST----------
df_original = pd.read_csv('dep1.csv')

#----------CALCULATE EPOCH TIMESTAMP----------
def ep_ts(i, ts):
	if i <= ts:
		return ts;
	if i > ts:
		return ep_ts(i, (ts + 86400))

#----------INSERT EPOCH TIMESTAMPS INTO ONE LIST----------
for index in df_original.ts_second:
	ts_ep.insert(index,(ep_ts(index, 1485714600))) #Num is the original timestamp

#----------INSERT TIME ELAPSED (SECONDS) OF DAY IN LIST----------
for index in df_original.ts_second:
	ts_est.insert(index, (df_original['ts_second'].tolist()[i] - (ts_ep[i] - 86400)))
	i = i + 1

#----------JOIN 2 LISTS INTO DATAFRAME----------
df_original['ts_epoch'] = pd.DataFrame({'ts_epoch': ts_ep})
df_original['ts_est'] = pd.DataFrame({'ts_est': ts_est})

#----------CALCULATE BUCKET IDS----------
def calc_bucket(x):
	return math.ceil((float(x['ts_est'])/86400.0)*144.0)

#----------CALCULATE DATES----------
ts_date = df_original['ts_second'].tolist()
for j, val in enumerate(ts_date):
	ts_date[j] = datetime.datetime.fromtimestamp(ts_date[j]).strftime("%B %d, %Y")
df_original['date'] = pd.DataFrame({'ts_date': ts_date})
	
#----------CALCULATE DAYS OF THE WEEK----------
def sec_to_day(x):
	return datetime.date.fromtimestamp(x).strftime('%A')

ts_day = df_original['ts_second'].tolist()
for j, val in enumerate(ts_day):
	ts_day[j] = sec_to_day(ts_day[j])

df_original['day'] = pd.DataFrame({'ts_day': ts_day})

#----------ADD BUCKET IDS TO DATAFRAME----------
df_original['bucket_id'] = df_original.apply(calc_bucket, axis=1)

#----------SPLIT INTO WEEKEND AND WEEKDAY----------
mask = df_original['day'].isin(['Saturday','Sunday'])

df_weekend = df_original[mask]
df_weekday = df_original[~mask]

#----------CREATE DATAFRAMES FOR TRAINING AND FOR TESTING----------
mask = df_original['ts_second'] <= int(datetime.datetime( 2017, 5, 30, 0, 0, 0 ).strftime('%s')) #yyyymmdd hhmmss
  
df_training_set = df_original[mask]
df_testing_set = df_original[~mask]   

#----------ALL DATA VARIABLES----------
z = 1 #Bucket ID
x = 0 #Column/Row Toggle
data_field = 'Mean'
row_index = 'Rvolt'
rel_col = ['bucket_id',row_index]

def get_results(df): #Add metrics 
	df['Mean'] = df_bucket_data.mean(axis=x)
	df['StanDev'] = df_bucket_data.std(axis=x)
	df['Minimum'] = df_bucket_data.min(axis=x)
	df['Maximum'] = df_bucket_data.max(axis=x)
	df['Median'] = df_bucket_data.median(axis=x)
	df['Sum'] = df_bucket_data.sum(axis=x)
	df['Count'] = df_bucket_data.count(axis=x)
	df['Bucket'] = z
	return df

def drop_day(df_source): #delete time data
	df_source = df_source.drop(['bucket_id','ts_est','ts_epoch','ts_second','date','day'], axis=1)

for col in rel_col:
	df_bucket_data = df_training_set[df_training_set['bucket_id'] == z]
	df_bucket_data_test = df_testing_set[df_testing_set['bucket_id'] == z]
	df_bucket_data_end = df_weekend[df_weekend['bucket_id'] == z]
	df_bucket_data_day = df_weekday[df_weekday['bucket_id'] == z]
	
buckets = range(1, 145)

df_result = pd.DataFrame(index=df_bucket_data.columns) #Create result DFs
df_result_test = pd.DataFrame(index=df_bucket_data_test.columns)
df_result_weekend = pd.DataFrame(index=df_bucket_data_end.columns)
df_result_weekday = pd.DataFrame(index=df_bucket_data_day.columns)

for b in buckets: #Separate the result data for test, training, weekend & weekday
	z = b
	df_bucket_data = df_original[df_original['bucket_id'] == z]
	df_bucket_data_test = df_original[df_original['bucket_id'] == z]
	df_bucket_data_end = df_weekend[df_weekend['bucket_id'] == z]
	df_bucket_data_day = df_weekday[df_weekday['bucket_id'] == z]
	
	drop_day(df_bucket_data) #Drop all time-sensitive data
	drop_day(df_bucket_data_test)
	drop_day(df_bucket_data_end)
	drop_day(df_bucket_data_day)
	
	df_result = (df_result.append(get_results(pd.DataFrame(index=df_bucket_data.columns)))).dropna(how='all')
	df_result_test = (df_result.append(get_results(pd.DataFrame(index=df_bucket_data_test.columns)))).dropna(how='all')
	df_result_weekend = (df_result.append(get_results(pd.DataFrame(index=df_bucket_data_end.columns)))).dropna(how='all')
	df_result_weekday = (df_result.append(get_results(pd.DataFrame(index=df_bucket_data_day.columns)))).dropna(how='all')

def normaliser(val): #Calculate the normalised value
	avg = float((df_result.loc[df_result.index.isin([row_index]) & df_result['Bucket'].between(1, 144) , 'Mean']).iloc[0])
	std = float((df_result.loc[df_result.index.isin([row_index]) & df_result['Bucket'].between(1, 144) , 'StanDev']).iloc[0])
	
	normalised = ((val - avg) / std)
	return normalised

random_date_list = list(np.random.choice(list(df_testing_set['date'].unique()),8)) #list of *  8  * unique dates from the entire list of data

df_new = df_testing_set[df_testing_set['date'].isin(random_date_list)] #New DataFrames for graphing
df_normalised = pd.DataFrame()

for date1 in random_date_list: #For each unique data in the random date list
	df_new = df_testing_set[df_testing_set['date'] == date1]
	title = date1
	
	if df_new.iloc[0]['day'] in ['Saturday', 'Sunday']: #If weekend
		df_shader = df_result_weekend.copy()
		title += " - Weekend"
	else: #If weekday
		df_shader = df_result_weekday.copy()
		title += " - Weekday"
	
	df_shader['Upper'] = (df_shader[data_field] + df_shader['StanDev']) #Storing primary boundary data in df
	df_shader['Lower'] = (df_shader[data_field] - df_shader['StanDev'])
	
	y_axis = df_shader.loc[df_shader.index.isin([row_index]) & df_shader['Bucket'].between(1, 144), data_field].tolist() #Y-Axis and secondary boundaries
	minus = df_shader.loc[df_shader.index.isin([row_index]) & df_shader['Bucket'].between(1, 144), 'Lower'].tolist()
	plus = df_shader.loc[df_shader.index.isin([row_index]) & df_shader['Bucket'].between(1, 144), 'Upper'].tolist()
	del y_axis[-1], minus[-2], plus[-2]
	
	all_avg = df_shader.loc[row_index, data_field].tolist() #Averages and Standard Deviations needed to calculate normalised values
	all_stan = df_shader.loc[row_index, 'StanDev'].tolist()
	del all_avg[-1], all_stan[-1]
	
	norms = [] #Store all normalised values in this list
		norms.append(normaliser(item))
	for item in y_axis:

	df_normalised = df_normalised.append(pd.DataFrame({'Values': y_axis, 'Averages': all_avg, 'StanDev': all_stan, 'Normalised': norms})) #Normalisation data
	standevs = df_shader.loc[df_shader.index.isin([row_index]) & df_shader['Bucket'].between(1, 144), 'StanDev'].tolist() #Standard Deviations
	del standevs[-1]
	
	lower_bound = np.array(y_axis) - np.array(standevs) #Boundaries
	upper_bound = np.array(y_axis) + np.array(standevs)

	plt.title(title) #Axis Label Data
	plt.xlabel("Time of Day (10m Intervals)")
	plt.ylabel("Normalised Values [Anomaly Score] for " + row_index)
	ax2 = plt.twinx()
	ax2.set_ylabel(data_field + " values for " + row_index)
	
	ax2.axhline(y=max(upper_bound.tolist()), color='red') #Intercepts
	ax2.axhline(y=min(lower_bound.tolist()), color='red')
	
	ax2.fill_between(buckets, upper_bound, (np.array(plus) + np.array(standevs)), color='orange') #Colour fills
	ax2.fill_between(buckets, lower_bound, upper_bound, color='lightgreen')
	ax2.fill_between(buckets, (np.array(minus) - np.array(standevs)), lower_bound, color='orange')
	
	plt.plot(range(0, len(df_new[row_index].tolist())),df_new[row_index].tolist()) #Plot X,Y
	norms = norms[:len(range(0, len(df_new[row_index].tolist())))] #Normalised Values
#	plt.plot(list(compress(buckets, list(starmap(gt, zip(y_axis, list(upper_bound)))))), list(compress(y_axis, list(starmap(gt, zip(y_axis, list(upper_bound)))))) , 'kx') #Marking the anomalies
#	plt.plot(list(compress(buckets, list(starmap(lt, zip(y_axis, list(upper_bound)))))), list(compress(y_axis, list(starmap(lt, zip(y_axis, list(upper_bound)))))), 'kx')
	plt.show()
	
	
	
	
	
	
	
#del df_original, df_testing_set, z, df_result, df_result_weekend, df_result_weekday, x, random_date_list, df_new, df_normalised, date1
#del title, y_axis, minus, plus, all_avg, all_stan, norms, standevs, lower_bound, upper_bound, plt, ax2, row_index
#del df_weekend, df_weekday, df_bucket_data, df_bucket_data_test, df_bucket_data_end, df_bucket_data_day, df_result_test
#del col, rel_col, df_training_set, ts_day, j, val, mask, ts_date, i, index, ts_ep, ts_est, b, buckets, data_field, item
