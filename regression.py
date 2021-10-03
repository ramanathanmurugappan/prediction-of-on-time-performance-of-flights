import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

# Load the raw data from CSV file into a DataFrame
raw_data_path = "/home/inithan/Desktop/git/movie/flight_data.csv"
data = pd.read_csv(raw_data_path, index_col=0)

#flights that have only an arrival delay
data=data[data['ArrDel15']==1]


# catatagorize the variables from EDA
data = data.drop(['CRSDepTime_time_stamp_round',
                  'CRSArrTime_time_stamp_round',
                 'weather_time_stamp_origin',
                  'weather_time_stamp_dest', 'Origin', 'Dest', 'airport_dest', 'Flights', 'airport_origin'], axis=1)

catagorical_varible = ['winddir16Point_origin',
                       'winddir16Point_dest',
                       'weatherCode_origin',
                       'weatherCode_dest']

numerical_variable = ['Distance',
                      'DistanceGroup',
                      'CRSElapsedTime',
                      'windspeedKmph_origin',
                      'FeelsLikeF_origin',
                      'DewPointF_origin',
                      'HeatIndexF_origin',
                      'cloudcover_origin',
                      'precipMM_origin',
                      'pressure_origin',
                      'WindGustKmph_origin',
                      'visibility_origin',
                      'tempF_origin',
                      'WindChillF_origin',
                      'winddirDegree_origin',
                      'humidity_origin',
                      'windspeedKmph_dest',
                      'FeelsLikeF_dest',
                      'DewPointF_dest',
                      'HeatIndexF_dest',
                      'cloudcover_dest',
                      'precipMM_dest',
                      'pressure_dest',
                      'WindGustKmph_dest',
                      'visibility_dest',
                      'tempF_dest',
                      'WindChillF_dest',
                      'winddirDegree_dest',
                      'humidity_dest']

target_variable = ['ArrDelayMinutes']
data = data[catagorical_varible+numerical_variable+target_variable]
data = pd.get_dummies(data, columns=catagorical_varible)
Target = data[target_variable]
#data=data.drop(catagorical_varible, axis=1)
data = data.drop(target_variable, axis=1)

"""
##############################################################
REGRESSION
1. LinearRegression
2. Gradient Boosting Regressor
3. Extra Trees Regressor

##############################################################
"""
# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data, Target, test_size=0.2, random_state=69)

# function to get print mertics for each algorithm
def model_accuracy(model, X_train, X_test, y_train, y_test):
    print("*"*10)
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MAE:\t",mean_absolute_error(y_test, y_pred))
    print("RMSE:\t",np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:\t",r2_score(y_test, y_pred))
    print("*"*10)

model_accuracy(GradientBoostingRegressor(), X_train, X_test, y_train, y_test)
model_accuracy(ExtraTreesRegressor(n_estimators=200,n_jobs=-1,verbose=True), X_train, X_test, y_train, y_test)
model_accuracy(LinearRegression(), X_train, X_test, y_train, y_test)