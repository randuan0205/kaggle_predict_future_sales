# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 07:18:01 2019

@author: randu
"""
### import libraries
import os
import numpy as np
import pandas as pd
import datetime as dt
import time
import sys
from itertools import product
#%matplotlib inline
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from xgboost import plot_importance

import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
sys.version_info

##mark this is an updated file

### import data                                      
os.chdir('C:\\Users\\randu\\OneDrive\\Documents\\datascience\\Python Scripts\\predict_sales\\final_doc')
#os.chdir('your directory')
items = pd.read_csv('items.csv')                   
shops = pd.read_csv('shops.csv')                   
cats = pd.read_csv('item_categories.csv')          
train = pd.read_csv('sales_train.csv.gz')          
test = pd.read_csv('test.csv.gz').set_index('ID')

###get rid of outliers
sns.boxplot(train.item_cnt_day)  ## sales unit's boxplot
plt.show()
train = train[train.item_cnt_day<=1000]   ##get rid of outliers for sales
train.item_cnt_day=train.item_cnt_day.apply(lambda x: x if x>0 else 0)  ##convert negative sales to 0

sns.boxplot(train.item_price)  ## price boxplot
plt.show()
train = train[train.item_price<50000] ## correct outliers for item price
correct_price = train[(train.date_block_num ==4)&(train.shop_id ==32)&(train.item_id ==2973)&(train.item_price >0)].item_price.mean()
train.loc[train.item_price<=0, 'item_price'] = correct_price

###Several shops are duplicates of each other (according to its name). Fix train and test set.
#### Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
#### Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
#### Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

###step0: build a helper function to downcast datatype:
#### note: this step can help decrease data size a lot.
#### the solution is provide by this notebook: https://www.kaggle.com/kyakovlev/1st-place-solution-part-1-hands-on-data
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

### STEP1: create dim matrix, aggregate sales by month, combine dims with monthly sales, combine train with test
def data_process_1 ():
    # create a list of all dimensions, will make up some dimensions with 0 since some of items in test don't exist in training data
    ##note: the method is inspired by this notebook: https://www.kaggle.com/dlarionov/feature-engineering-xgboost
    dims = pd.DataFrame ()
    periodcnt = len(train['date_block_num'].unique())
    for i in range(periodcnt):
        data_segment = train[train.date_block_num == i]
        dim_df = pd.DataFrame(list(product([i], data_segment.shop_id.unique(), data_segment.item_id.unique())))
        dims = dims.append(dim_df)
    dims.columns = ['date_block_num','shop_id','item_id']
    print("dims df's shape: {}".format(dims.shape))
    
    # generate monthly total sales
    train2 = train[['date_block_num','shop_id','item_id','item_cnt_day']].\
             groupby(['date_block_num','shop_id','item_id'], as_index = False).\
             sum()        # shape(1,609,124, 4)
    print("monthly aggregated training data's dimension: {}".format(train2.shape))

    # combine dims with monthly sales:
    fulldata = pd.merge(dims, train2, on=['date_block_num','shop_id','item_id'], how='left')
    print("combined df's shape: {}".format(fulldata.shape))
    fulldata.sort_values (by = ['item_id', 'shop_id', 'date_block_num'], inplace = True)
    
    #combine train with test 
    test['date_block_num'] = 34
    maindata = pd.concat([fulldata, test], ignore_index = True)  # shape(1,823,324, 4) 
    maindata.columns = ['date_block_num', "item_cnt", "item_id", "shop_id"]
    maindata['item_cnt'] = (maindata['item_cnt']
                                    .fillna(0)   # file NAs with zeros
                                    .clip(0,20)) # clip target based on instruction
    
    print("final output data's dimension: {}".format(maindata.shape))
    maindata = downcast_dtypes(maindata)
    return maindata

ts = time.time()
maindata = data_process_1 ()
time.time() - ts

### STEP2: append year and month to dataset 
def data_process_2 ():
    ## append year-month to the dataframe
    train['date'] = train['date'].str.split('.').map(lambda x: x[2]+'-'+x[1])
    train['date'] = pd.to_datetime(train.date,format = '%Y-%m')
    datedf = pd.DataFrame(train.date.unique())
    timenumdf = pd.DataFrame(train.date_block_num.unique())
    datedf= datedf.append( [pd.to_datetime('2015-11',format = '%Y-%m')])
    timenumdf = timenumdf.append( [34.0])
    timedf = pd.concat([datedf, timenumdf], axis=1)
    timedf.columns = ['time_period','date_block_num']
    maindata2 = pd.merge(maindata, timedf, how = 'left')
    
    maindata2['year'] = maindata2['time_period'].dt.year
    maindata2['month'] = maindata2['time_period'].dt.month
    maindata2.drop('time_period', axis=1, inplace = True)
    print ('data output dim:{}'.format(maindata2.shape))
    return maindata2

ts = time.time()
maindata2 = data_process_2 ()
del maindata
time.time() - ts

### STEP3: clean up categorical features, combine with main data:
def data_process_3 (df = maindata2, shop_data = shops, cat_data = cats, item_data = items):
    shop_data.loc[shop_data.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    #Each shop_name starts with the city name.
    shop_data['city'] = shop_data['shop_name'].str.split(' ').map(lambda x: x[0])     # 32 unique cities
    shop_data.loc[shop_data.city == '!Якутск', 'city'] = 'Якутск'
    shops = shop_data[['shop_id','city']]
    
    #Each category contains type and subtype in its name.
    cat_data['split'] = cat_data['item_category_name'].str.split('-')
    cat_data['type'] = cat_data['split'].map(lambda x: x[0].strip())                  # 20 unique types
    # if subtype is nan then type
    cat_data['subtype'] = cat_data['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip()) # 65 unique subtypes
    cats = cat_data[['item_category_id','type', 'subtype']]
    
    # clean up items data
    items = item_data.drop(['item_name'], axis=1)    ##provide item_id and category id

    #combine metrics with dims
    cat_metric = df.merge(shops, on=['shop_id'], how='left').\
                  merge(items, on=['item_id'], how='left').\
                  merge(cats, on=['item_category_id'], how='left')
    return cat_metric

ts = time.time()
cat_metric = data_process_3 ()
del maindata2
del shops
del cats
del items
time.time() - ts

### STEP4: create a function to create lagging metrics for time series data
#### note: tried pandas shift function before, but found using pandas merge func is much more efficient
def feature_timeseries(df, lag, col = 'item_cnt'):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    
    for i in lag:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = df.merge(shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df.fillna (0, inplace = True)
    return df

ts = time.time()
lags = [1,2,3,4,5,6]
data_timeseries = feature_timeseries (df = cat_metric, lag = lags, col = 'item_cnt')
del cat_metric
del lags
(time.time() - ts)

### STEP 5 create the feature of sales growth rate
#### note: based on EDA analysis,the items sales have big fluctuation caused by seasonality. Understand the latest sales trend is very important for prediction.
##create the columns for bsline_sls
ts=time.time()
bsline_sls = data_timeseries.iloc[:,11:16] ## use previous 5 mths' sales to create the baseline
data_timeseries['bsline_sls_avg'] = pd.Series(np.mean(bsline_sls,axis=1))

##create the function to calc growth rate
def sales_growth_ratio (lag1, avg):
    if lag1 == avg:
        grow_rate = 0
    elif avg == 0:
        grow_rate = 1
    else:
        grow_rate = (lag1-avg)/avg
    return grow_rate

##create the growth rate feature
ts=time.time()    
data_timeseries['sls_growth_rate']= data_timeseries.apply(lambda x: sales_growth_ratio(x['item_cnt_lag_1'],x['bsline_sls_avg']), axis=1)
data_timeseries.drop(['bsline_sls_avg'], axis=1, inplace=True)
del bsline_sls
print(time.time() - ts)
print ('finished step 5 at {}'.format(dt.datetime.now()))

##step 6: create sales related features - recency (time of the most recent sales) / frequency/ the time of the first sales
# generate monthly sales
sales_data = train[['date_block_num','shop_id','item_id','item_cnt_day']].\
             groupby(['date_block_num','shop_id','item_id'], as_index = False).sum()   
sales_data2 = sales_data[sales_data.item_cnt_day>0]  ##only keep the record with valid sales
sales_data2.rename(columns = {'item_cnt_day':'item_cnt_mth'}, inplace = True)

###6.1 create freq_item_shop
freq_item_shop = pd.DataFrame ()
for block_num in range(6,35):
    start = block_num-6
    end = block_num -1
    tempdata=sales_data2[(sales_data2.date_block_num>=start)&(sales_data2.date_block_num<=end)]
    tempdata2 = tempdata.groupby(['shop_id', 'item_id'])[['date_block_num']].count()
    tempdata2.columns = ['freq']
    tempdata2.reset_index(inplace=True)
    tempdata2['date_block_num'] = block_num
    tempdata2['freq_item_shop']=tempdata2['freq']/6
    freq_item_shop = freq_item_shop.append(tempdata2)
freq_item_shop.drop('freq', axis=1, inplace=True)
#merge with main dataset
data_timeseries = data_timeseries.merge(freq_item_shop, how='left', on=['shop_id','item_id','date_block_num']).\
                                  fillna(0)
del freq_item_shop
                                  
###6.2 create last_sls_item_shop
last_item_shop = pd.DataFrame ()
for block_num in range(6,35):
    tempdata=sales_data2[sales_data2.date_block_num<block_num]
    tempdata2 = tempdata.groupby(['shop_id', 'item_id'])[['date_block_num']].max()
    tempdata2.columns = ['last_sls_item_shop']
    tempdata2.reset_index(inplace=True)
    tempdata2['date_block_num'] = block_num
    last_item_shop = last_item_shop.append(tempdata2)
#merge with main dataset
data_timeseries = data_timeseries.merge(last_item_shop, how='left', on=['shop_id','item_id','date_block_num'])
##calc time delta
data_timeseries['last_sls_item_shop'] = data_timeseries['date_block_num'] - data_timeseries['last_sls_item_shop']
##replace NA with a big value
data_timeseries.last_sls_item_shop.fillna(999999, inplace=True)
del last_item_shop

##6.3 create first_sls_item_shop
first_item_shop = sales_data2.groupby(['shop_id', 'item_id'])[['date_block_num']].min()
first_item_shop.columns = ['first_sls_item_shop']
first_item_shop.reset_index(inplace=True)
#merge with main dataset
data_timeseries = data_timeseries.merge(first_item_shop, how='left', on=['shop_id','item_id'])
##calc time delta
data_timeseries['first_sls_item_shop'] = data_timeseries['date_block_num'] - data_timeseries['first_sls_item_shop']
##replace NA
data_timeseries.first_sls_item_shop.fillna(999999, inplace=True)
##need to replace negative values with 99999999
data_timeseries.loc[data_timeseries.first_sls_item_shop<=0,'first_sls_item_shop']=999999
del first_item_shop
print ('finished step 6.3 at {}'.format(dt.datetime.now()))

###6.4 create freq_item
freq_item = pd.DataFrame ()
for block_num in range(6,35):
    start = block_num-6
    end = block_num -1
    tempdata=sales_data2[(sales_data2.date_block_num>=start)&(sales_data2.date_block_num<=end)]
    tempdata2 = tempdata[['item_id','date_block_num']].drop_duplicates()
    tempdata3 = tempdata2.groupby(['item_id'])[['date_block_num']].count()
    tempdata3.columns = ['freq_item']
    tempdata3.reset_index(inplace=True)
    tempdata3['date_block_num'] = block_num
    tempdata3['freq_item']=tempdata3['freq_item']/6
    freq_item = freq_item.append(tempdata3)
#merge with main dataset
data_timeseries = data_timeseries.merge(freq_item, how='left', on=['item_id','date_block_num']).\
                                  fillna(0)
del freq_item

###6.5 create last_sls_item
last_item = pd.DataFrame ()
for block_num in range(6,35):
    tempdata=sales_data2[sales_data2.date_block_num<block_num]
    tempdata2 = tempdata.groupby(['item_id'])[['date_block_num']].max()
    tempdata2.columns = ['last_sls_item']
    tempdata2.reset_index(inplace=True)
    tempdata2['date_block_num'] = block_num
    last_item = last_item.append(tempdata2)
#merge with main dataset
data_timeseries = data_timeseries.merge(last_item, how='left', on=['item_id','date_block_num'])
##calc time delta
data_timeseries['last_sls_item'] = data_timeseries['date_block_num'] - data_timeseries['last_sls_item']
##replace NA
data_timeseries.last_sls_item.fillna(999999, inplace=True)
del last_item

###6.6 create first_sls_item
first_item = sales_data2.groupby(['item_id'])[['date_block_num']].min()
first_item.columns = ['first_sls_item']
first_item.reset_index(inplace=True)
#merge with main dataset
data_timeseries = data_timeseries.merge(first_item, how='left', on=['item_id'])
##calc time delta
data_timeseries['first_sls_item'] = data_timeseries['date_block_num'] - data_timeseries['first_sls_item']
##replace NA
data_timeseries.first_sls_item.fillna(999999, inplace=True)
##need to replace negative values with 99999999
data_timeseries.loc[data_timeseries.first_sls_item<=0,'first_sls_item']=999999
del first_item

del [sales_data, sales_data2, tempdata, tempdata2, tempdata3, start, end, block_num]
print ('finished step 6 at {}'.format(dt.datetime.now()))

### STEP 7: use mean encoding to create new features
mldata = data_timeseries
del data_timeseries

## set up the function for mean-encode-avg
def mean_encode_avg (df = mldata, cat_cols = None, avg_lag = [1,2,3,4]):
    # define mean encoded features' names
    mean_col_name = 'mean_encode_avg_'
    for i in range(len(cat_cols)):
        mean_col_name = mean_col_name + '_' + cat_cols[i]
    # generate new columns for mean & std
    sum_metric = df.groupby(cat_cols)[['item_cnt']].agg(['mean'])
    sum_metric.columns = [mean_col_name]
    sum_metric.reset_index(inplace = True)
    #merge new data back to original dataset
    df = df.merge(sum_metric, on=cat_cols, how='left')
    ##create new data's time series data, and delelete the data columns for the current months
    df = feature_timeseries (df = df, lag = avg_lag, col = mean_col_name)   
    df.drop([mean_col_name], axis=1, inplace=True)
    return df

## 7.1 Create mean encoded features for city&item_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num', 'city', 'item_id'])
print(time.time() - ts)

## 7.2 Create mean encoded features for item_id 
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','item_id'])
print(time.time() - ts)

## 7.3 Create mean encoded features for category_id&shop_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','item_category_id', 'shop_id'])
print(time.time() - ts)
print ('finished step 7.3 at {}'.format(dt.datetime.now()))

## 7.4 Create mean encoded features for subtype&shop_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num', 'subtype', 'shop_id'], avg_lag = [1])
print(time.time() - ts)

## 7.5 Create mean encoded features for city&subtype
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num', 'city', 'subtype'], avg_lag = [1])
print(time.time() - ts)

## 7.6 Create mean encoded features for city&category
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num', 'city', 'item_category_id'], avg_lag = [1])
print(time.time() - ts)

## 7.7 Create mean encoded features for month
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num'], avg_lag = [1])
print(time.time() - ts)
print ('finished step 7.7 at {}'.format(dt.datetime.now()))

## 7.8 Create mean encoded features for type&shop_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num', 'type', 'shop_id'], avg_lag = [1])
print(time.time() - ts)

## 7.9 Create mean encoded features for shop_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','shop_id'], avg_lag = [1,2])
print(time.time() - ts)

## 7.10 Create mean encoded features for category_id
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','item_category_id'], avg_lag = [1])
print(time.time() - ts)
print ('finished step 7.10 at {}'.format(dt.datetime.now()))

## 7.11 Create mean encoded features for subtype
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','subtype'], avg_lag = [1])
print(time.time() - ts)

## 7.12 Create mean encoded features for type
ts=time.time()
mldata = mean_encode_avg (df = mldata, cat_cols = ['date_block_num','type'], avg_lag = [1])
print(time.time() - ts)
print ('finished step 7 at {}'.format(dt.datetime.now()))

###STEP 8 mldata feature preprocessing
mldata['item_id'] = LabelEncoder().fit_transform(mldata['item_id'])
mldata['shop_id'] = LabelEncoder().fit_transform(mldata['shop_id'])
mldata['city'] = LabelEncoder().fit_transform(mldata['city'])
mldata['item_category_id'] = LabelEncoder().fit_transform(mldata['item_category_id'])
mldata['type'] = LabelEncoder().fit_transform(mldata['type'])
mldata['subtype'] = LabelEncoder().fit_transform(mldata['subtype'])

mldata = downcast_dtypes(mldata)
print(mldata.info())
print ('finished step 8 data preprocessing at {}'.format(dt.datetime.now()))

###STEP 9 Model training (random forest / Gradient Boosting Tree / Linear Regression)
print ('start data_splitting at {}'.format(dt.datetime.now()))
#### note: given that this is a time-series forecasting project, the validation set should be right before test set.
mldata = mldata.loc[mldata.date_block_num>5,:]
X_train = mldata[mldata.date_block_num < 33].drop(['item_cnt'], axis=1)
y_train = mldata[mldata.date_block_num < 33]['item_cnt']
X_dev = mldata[(mldata.date_block_num == 33)].drop(['item_cnt'], axis=1)
y_dev = mldata[(mldata.date_block_num == 33)]['item_cnt']
X_test = mldata[mldata.date_block_num == 34].drop(['item_cnt'], axis=1)

##save data set with all features
tuple_objects = (mldata)
pkl_filename = "C:\\Users\\randu\\OneDrive\\Documents\\datascience\\Python Scripts\\predict_sales\\final_doc\\mldata.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tuple_objects, file, protocol = 4)
del mldata
del train
#####################step 9.1.1 rf tuning - n_estimators##################################################
print ('start random forest tuning for n_estimators at {}'.format(dt.datetime.now()))
ns=time.time()
#### step 1: fit a model with a big number of trees
rfmodel_tuning = RandomForestRegressor (max_depth=18, n_estimators=300, n_jobs = 6, max_features = 'log2', random_state=100)
rfmodel_tuning.fit(X_train, y_train)
print(time.time()-ns)

#### Step 2: Get predictions for each tree in Random Forest separately.
predictions = []
for tree in rfmodel_tuning.estimators_:
    predictions.append(tree.predict(X_dev).clip(0, 20))

#### Step 3: Concatenate the predictions to a tensor of size (number of trees, number of objects, number of classes).
predictions = np.vstack(predictions)

#### Step 4: Сompute cumulative average of the predictions. That will be a tensor, that will contain predictions of the random forests for each n_estimators.
cum_mean = np.cumsum(predictions, axis=0)/np.arange(1, predictions.shape[0] + 1)[:, None]

#### Step 5: Get accuracy scores for each n_estimators value
scores = []
for pred in cum_mean:
    scores.append(np.sqrt(mean_squared_error(y_dev, pred)))

####figure plotting    
plt.figure(figsize=(10, 6))
plt.plot(scores, linewidth=3)
plt.xlabel('num_trees')
plt.ylabel('rmse')

del (rfmodel_tuning, predictions, cum_mean, scores)
##the best choice is 198 trees

##################################step 9.1.2 rf tuning - n_estimatorsrandom forrest PARAMETER TUNING #################################################################
print ('start random forest tuning for more features at {}'.format(dt.datetime.now()))
rfmodel = RandomForestRegressor (max_depth=18, n_estimators=110, n_jobs = 6, max_features = 'log2', random_state=100)
rfmodel.fit(X_train, y_train)

print ('start rf predicting at {}'.format(dt.datetime.now()))
predict_dev_rf = rfmodel.predict(X_dev).clip(0, 20)
print("Root Mean squared error: {}".format(np.sqrt(mean_squared_error(y_dev, predict_dev_rf))))
predict_train_rf = rfmodel.predict(X_train).clip(0,20)
print("Root Mean squared error (on training data): {}".format(np.sqrt(mean_squared_error(y_train, predict_train_rf))))

##### predict test set
predict_test_rf = rfmodel.predict(X_test).clip(0, 20)
## check feature importance
importance = rfmodel.feature_importances_
importance = pd.DataFrame(importance, index=X_train.columns, columns=["Importance"])
importance.sort_values(by=['Importance'], ascending = False, inplace = True)
importance['Importance'].nlargest(15).plot(kind='barh')
plt.show()
print(time.time() - ts)

##log of performance (100 trees): train: 0,700 valid: 0.926 public: 0.885540 private: 0.897927
##log of performance (198 trees): train: 0,699 valid: 0.920 public: 0.885670 private: 0.898222
##log of performance (110 trees): train: 0,700 valid: 0.9258 public: 0.884946 private: 0.897573
##conclusion: 110 trees is the best solution

################################step 9.2 XGB model Parameter Tuning###############################################
print ('start Gradient Boosting Tree tuning at {}'.format(dt.datetime.now()))
ts = time.time()
## set up a function for model training:
def modelfit(alg):
    #Fit the algorithm on the data
    alg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_dev, y_dev)],
            eval_metric='rmse',
            verbose=True,
            early_stopping_rounds = 75)    
    #Predict dev set:
    predict_dev_xgb = alg.predict(X_dev).clip(0, 20)
    print("Root Mean squared error for prediction of dev: {}".format(np.sqrt(mean_squared_error(y_dev, predict_dev_xgb))))
    #Predict training set:
    predict_train_xgb = alg.predict(X_train).clip(0, 20)
    print("Root Mean squared error for prediction of train: {}".format(np.sqrt(mean_squared_error(y_train, predict_train_xgb))))
    return alg, predict_dev_xgb

xgb7 = XGBRegressor( learning_rate = 0.01,
                     n_estimators = 2000,
                     max_depth = 10,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.9,
                     colsample_bytree = 0.9,
                     objective='reg:squarederror',
                     nthread = 5,
                     scale_pos_weight=1,
                     seed=27)

xgb_model, predict_dev_xgb = modelfit(xgb7)

##### predict test set
predict_test_xgb = xgb_model.predict(X_test).clip(0, 20)

#### check feature importance
plot_importance(xgb_model)

time.time() - ts
#### tuning results log:
##### learning rate 0.05 (xgb71) - results: 0.9124 / 0.8395 /n_estimator:377 /   time: 10861
##### learning rate 0.05 & gamma 0.2 (xgb71) - results: 0.9091 / 0.8359 /n_estimator:414 /   time: 13023
##### learning rate 0.05 & gamma 0.5 (xgb71) - results: 0.9091 / 0.8359 /n_estimator:414 /   time: 13023
##### learning rate 0.05 & gamma 1 (xgb71) - results: 0.9123 / 0.8412 /n_estimator:353 /   time: 10343
##### learning rate 0.01 & gamma 1 (xgb71) - results: 0.9161 / 0.8568 /n_estimator:1106 /   time: 25472
##### learning rate 0.01 & gamma 0.2 (xgb71) - results: 0.9158 / 0.8566 /n_estimator:1034 /   time: 25312
##### learning rate 0.01 & gamma 0.2 & max_depth 10 & subsample 0.9 & colsample 0.9 (xgb71) - results: 0.908 / 0.8299 /n_estimator:233 /time: 13781
##### learning rate 0.01 & gamma 0.2 & max_depth 15 & subsample 0.9 & colsample 0.9 (xgb71) - results: 0.950 / 0.7261 /n_estimator:168 /time: 18539
##### final result: learning rate 0.01 & gamma 0.2 & max_depth 10 & subsample 0.9 & colsample 0.9 (xgb71) - 
   ##results: DEV:0.907 / TRAIN:0.834/ PUBLIC: 0.912725 and PRIVATE0.916400 /n_estimator:222 /time: 13533

################################step 9.3 Linear Regression training###############################################   
print ('start training for lr model at {}'.format(dt.datetime.now()))
ts = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
predict_dev_lr = lr_model.predict(X_dev).clip(0, 20)
print("Root Mean squared error for prediction of dev: {}".format(np.sqrt(mean_squared_error(y_dev, predict_dev_lr))))
predict_train_lr = lr_model.predict(X_train).clip(0, 20)
print("Root Mean squared error for prediction of train: {}".format(np.sqrt(mean_squared_error(y_train, predict_train_lr))))

##### predict test set
predict_test_lr = lr_model.predict(X_test).clip(0, 20)
print(time.time()-ts)

##performance log: dev: 0.965; train: 0.944; public: ; private:

###############################step 9.4 model stackig##########################################
print ('start training for model stacking at {}'.format(dt.datetime.now()))
ts = time.time()
stacked_predictions = np.column_stack((predict_dev_rf, predict_dev_xgb, predict_dev_lr))
stacked_test_predictions = np.column_stack((predict_test_rf, predict_test_xgb, predict_test_lr))
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_dev)

##predict test set
predict_test_meta = meta_model.predict(stacked_test_predictions)
print(time.time()-ts)

##performance log: dev: ; train: ; public: 0.908251; private:0.913162

####appendix: models' serialization########################################
##save models
tuple_objects = (rfmodel, xgb_model, lr_model, meta_model)
pkl_filename = "C:\\Users\\randu\\OneDrive\\Documents\\datascience\\Python Scripts\\predict_sales\\final_doc\\models_predict_sales.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tuple_objects, file, protocol = 4)
del (tuple_objects, pkl_filename)
    
##### load models 
pkl_file = open('models_predict_sales.pkl', 'rb')
rfmodel, xgb_model, lr_model, meta_model = pickle.load(pkl_file)
pkl_file.close()

#del (submission_xgb, xgb_model, predict_test_xgb, predict_dev_xgb)    



