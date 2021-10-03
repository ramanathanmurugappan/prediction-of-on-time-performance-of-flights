import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# Load the raw data from CSV file into a DataFrame
raw_data_path = "/home/inithan/Desktop/git/movie/flight_data.csv"
data = pd.read_csv(raw_data_path, index_col=0)


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

target_variable = ['ArrDel15']
data = data[catagorical_varible+numerical_variable+target_variable]
data = pd.get_dummies(data, columns=catagorical_varible)
Target_label = data[target_variable]
#data=data.drop(catagorical_varible, axis=1)
data = data.drop(target_variable, axis=1)


"""
##############################################################
CLASSIFICATION
1. no sampling
2. Random Undersampling
3. Random Oversampling
4. SMOTE Oversampling

##############################################################
"""
# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data, Target_label, test_size=0.2, random_state=69)


# function to get print mertics for each algorithm
def model_accuracy(model, X_train, X_test, y_train, y_test):
    print("*"*10)
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))
    print("*"*10)


# RAW MODELS -No Sampling
print("RAW MODELS -No Sampling Models Accuracies")
print("#"*20)
model_accuracy(RandomForestClassifier(), X_train, X_test, y_train, y_test)
model_accuracy(GaussianNB(), X_train, X_test, y_train, y_test)
model_accuracy(AdaBoostClassifier(), X_train, X_test, y_train, y_test)
model_accuracy(ExtraTreesClassifier(), X_train, X_test, y_train, y_test)
model_accuracy(GradientBoostingClassifier(), X_train, X_test, y_train, y_test)
model_accuracy(LogisticRegression(), X_train, X_test, y_train, y_test)
print("#"*20)


# Random Undersampling
rus = RandomUnderSampler(random_state=69)
X_sampled, y_sampled = rus.fit_resample(X_train, y_train)

print("Random Undersampling Models Accuracies")
print("#"*20)
model_accuracy(RandomForestClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GaussianNB(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(AdaBoostClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(ExtraTreesClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GradientBoostingClassifier(),X_sampled, X_test, y_sampled, y_test)
model_accuracy(LogisticRegression(), X_sampled, X_test, y_sampled, y_test)
print("#"*20)


# Random Oversampling
ros = RandomOverSampler(random_state=69)
X_sampled, y_sampled = ros.fit_resample(X_train, y_train)

print("Random Oversampling Models Accuracies")
print("#"*20)
model_accuracy(RandomForestClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GaussianNB(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(AdaBoostClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(ExtraTreesClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GradientBoostingClassifier(),X_sampled, X_test, y_sampled, y_test)
model_accuracy(LogisticRegression(), X_sampled, X_test, y_sampled, y_test)
print("#"*20)


# SMOTE Oversampling
smote = SMOTE(random_state=69)
X_sampled, y_sampled = smote.fit_resample(X_train, y_train)

print("SMOTE Oversampling Models Accuracies")
print("#"*20)
model_accuracy(RandomForestClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GaussianNB(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(AdaBoostClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(ExtraTreesClassifier(), X_sampled, X_test, y_sampled, y_test)
model_accuracy(GradientBoostingClassifier(),X_sampled, X_test, y_sampled, y_test)
model_accuracy(LogisticRegression(), X_sampled, X_test, y_sampled, y_test)
print("#"*20)
