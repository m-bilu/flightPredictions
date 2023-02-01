
## Importing all required libraries, remember to comment function of each library later
import pandas as pd ## R but in Python
import seaborn as sns ## Plots
import matplotlib.pyplot as plt ## more plots
import numpy as np # more linalg
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor # Training
from sklearn.model_selection import train_test_split, RandomizedSearchCV # Model Select
import math
import pickle


## Importing Flight Data
train_data = pd.read_excel('Data_Train.xlsx')
#print(len(train_data))
#print(train_data.head())

# Data Cleaning
print(train_data['Destination'].value_counts())
def clean(destination) :
    if destination == 'New Delhi':
        return 'Delhi'
    else:
        return destination

train_data['Destination'] = train_data['Destination'].apply(clean)

## Checking cleaned data
#print(len(train_data))
#print(train_data.head())
train_data.info()
## Route and Total Stops has some null values

# More datatype cleaning
# Creating columns for datetime values

# .dt takes a timeStamp object (similar to python builtin datetime object)
#   and returns array containing d, m, y
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.day
train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.month

# .drop() method removes specific column from train_data, return type is None
# Dont need this redundant but inefficient column anymore, data has been cleaned into
#   appropriate datatypes
train_data.drop('Date_of_Journey',inplace=True,axis=1)


### MORE CLEANING
# Same, but with arrival/departure hour, min
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
train_data.drop('Arrival_Time', inplace=True, axis=1)

train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
train_data.drop('Dep_Time', inplace=True, axis=1)

print(train_data.head())

### MORE CLEANING
# Droping duration column
duration = list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2: ## no minutes, split list is only 1 item
        if 'h' in duration[i]: ## if hour is in our time, then hour is nonzero
            duration[i] = duration[i] + ' 0m'

        else: # no hour, only minute, travel time less than 1 hr
            duration[i] = '0h ' + duration[i]

duration_hour = []
duration_min = []

for i in duration:
    h,m = i.split()
    duration_hour.append(int(h[:-1]))
    duration_min.append(int(m[:-1]))

train_data['Duration_hours'] = duration_hour
train_data['Duration_mins'] = duration_min

train_data.drop('Duration',axis=1,inplace=True)
train_data.head()

#### Last Cleaning Step: remove garbage columns:
train_data.drop(['Route', 'Additional_Info'],inplace=True,axis=1)

###### CLEANING DONE

## Plotting Airline vs Price
sns.catplot(x='Airline',y="Price",data=train_data.sort_values('Price',ascending=False),
kind='boxen',aspect=3,height=6)

#plt.show() # matplotlib used to print seaborn plot

#####
airline = train_data[['Airline']] ## [[]] outputs pandas dataframe instead of pandas series
airline = pd.get_dummies(airline,drop_first=True) 
## Dummy vars == Indicator vars, since this is categorical col

## Plotting Source vs Price
sns.catplot(x='Source',y='Price',data=train_data.sort_values('Price',ascending=False),
kind='boxen',aspect=3,height=4)

#plt.show() # matplotlib used to print seaborn plot

####
source = train_data[['Source']]
source = pd.get_dummies(source,drop_first=True)

# Same process with destination

## Plotting Destination vs Price
sns.catplot(x='Destination',y='Price',data=train_data.sort_values('Price',ascending=False),
kind='boxen',aspect=3,height=4)

#plt.show() # matplotlib used to print seaborn plot

####
destination = train_data[['Destination']] ## [[]] outputs pandas dataframe instead of pandas series
destination = pd.get_dummies(destination,drop_first=True)

#### Calculating # of stops per flight
train_data['Total_Stops'].replace({'non-stop':0,'1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}, inplace=True)
train_data.head()

## Checking Shapes
print(airline.shape) # .shape is like dim from R
print(source.shape)
print(destination.shape)
print(train_data.shape)

# Appropriate formats
# Combine indicator vectors into dataframe
data_train = pd.concat([train_data,airline,source,destination],axis=1)
data_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)
data_train.head()

# Removing train data from dataset
X = data_train.drop('Price',axis=1) # X contains dataframe without price, price is target
print(X.head())

y = data_train['Price']
y.head()

## NOW WE HAVE y = Beta * X, like STAT 331
## Let's draw correlation between diff variables in trainingData
## will help decide on model, interactions, etc
plt.figure(figsize=(10,10))
sns.heatmap(train_data.corr(),cmap='viridis',annot=True)

# plt.show()

# Total travel time is highly correlated with Duration Hours
# Price is correlated with total stops as well

#################################################

## Modelling Part
## Random Forest -> Research these methods

# Removing NaNs
for i in range(len(X['Total_Stops'])):
    if math.isnan(X['Total_Stops'][i]):
        X['Total_Stops'][i] = 0

reg = ExtraTreesRegressor()
reg.fit(X,y) # Fitting model like STAT 331

print(reg.feature_importances_)

## Checkign Feature Importance -> These are beta estimates
plt.figure(figsize = (12,8))
feat_importances = pd.Series(reg.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

##### By far, Total stops, journey day (weeknds) have largest impact on final price of airline ticket
#### Then, some airlines are much more expensive than others, but most have similar influence

########################################### SPLITTING OUR DATA
## DO MORE RESEARCH ON THIS!!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

########################################### TRAINING MODEL -> KFOLD?

# LOOK INTO THEORY OF RANDOM FOREST, RELATE TO REGULAR REGRESSION FROM STAT 331
numtrees = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

features = [1, 'sqrt']

depth = [int(x) for x in np.linspace(5, 30, num = 6)]

samples_split = [2,5,10,15,100]

samples_leaf = [1,2,5,10]

randomgrid = {'n_estimators': numtrees,
               'max_features': features,
               'max_depth': depth,
               'min_samples_split': samples_split,
               'min_samples_leaf': samples_leaf}

rf = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = randomgrid, 
scoring='neg_mean_squared_error', n_iter=10,cv=5,verbose=1,random_state=42,n_jobs=1)

rf.fit(X_train, y_train)

prediction = rf.predict(X_test)

## Plot residuals to measure accuracy of model
plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()
print('r2 score: ', sklearn.metrics.r2_score(y_test,prediction))
file = open('flightPredictions.pkl', 'wb')
pickle.dump(rf, file)

## NEXT STEP: Build a Flask app to turn data into application
