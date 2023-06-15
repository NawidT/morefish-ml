import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection as fs
from sklearn.tree import DecisionTreeRegressor
import pickle
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing, PolynomialFeatures
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

# Grab Datasets
ysi = pd.read_csv("../datasets/ysi_dataset.csv")
weather = pd.read_csv("../datasets/weather_dataset.csv")
ysi['datetime'] = pd.to_datetime(ysi['datetime']).dt.round('15min')
weather['datetime'] = pd.to_datetime(weather['datetime'])
combined = pd.merge(ysi, weather, on='datetime')
combined.reset_index(drop=True, inplace=True)
combined = combined.drop(columns=['experimentid_y'])

# Drop NaNs
#combined = combined.drop(combined[combined['ph'] < 5].index)
#combined = combined.drop(combined[combined['ph'] > 9].index)
combined = combined.drop(combined[combined['dissolved_oxygen_mg_l'] < 2].index)
combined = combined.drop(combined[combined['dissolved_oxygen_mg_l'] > 20].index)
combined = combined.drop(combined[combined['dissolved_oxygen_mg_l'] < 3/100 * combined['global_light_energy_w_m2']].index)
combined = combined.drop(combined[combined['dissolved_oxygen_mg_l'] >( 3/100 * combined['global_light_energy_w_m2']) + 15].index)


# Create Tests
test = combined.copy(True)
X_train,X_test,y_train,y_test = train_test_split(test[['ph', 'temperature_oc', 'global_light_energy_w_m2', 'humid_rh', 'airtemp_oc']], 
        test['dissolved_oxygen_mg_l'], test_size=0.15)

def stats(y_pred_all, y_test_all):
    # Calculate the R2 score
    r2 = r2_score(y_test_all, y_pred_all)

    print(f"R2 Score: {r2:.4f}")
    # Calculate MAE
    mae = mean_absolute_error(y_test_all, y_pred_all)

    # Calculate RMSE
    mse = mean_squared_error(y_test_all, y_pred_all, squared=False)

    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", math.sqrt(mse))

print("------------ Bagging Results ------------")
model = make_pipeline(preprocessing.SplineTransformer(), BaggingRegressor(n_jobs=5))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats(y_pred, y_test)

print("------------ Stacking Results ------------")
model = StackingRegressor(estimators=[('svr', LinearSVR(random_state=42)), ('rf', RandomForestRegressor(n_estimators=10,random_state=42, n_jobs=8))] )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats(y_pred, y_test)

print("------------ Random Forest Results ------------")
rf_regressor = make_pipeline(preprocessing.SplineTransformer(), RandomForestRegressor(n_estimators=24, max_depth=20))
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)
stats(y_pred, y_test)

print("------------ Poly Regression Results ------------")
poly_features = PolynomialFeatures(degree=5, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)
# Predict the target variable for training and test sets
y_pred = model.predict(X_test_poly)
stats(y_pred, y_test)

print("------------ Decision Tree Results ------------")
model = DecisionTreeRegressor(max_depth=24)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats(y_pred, y_test)

print("------------ Ada Boost Results ------------")
model = AdaBoostRegressor(learning_rate=0.01, loss='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats(y_pred, y_test)

print("------------ XG Boost Results ------------")
model = GradientBoostingRegressor(learning_rate=0.1, loss='huber', max_depth=5, criterion='squared_error')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats(y_pred, y_test)