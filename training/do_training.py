import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
import sklearn.preprocessing as preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

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
    return (r2, mae, mse)

def training_update_loop(filename):
    # Load the data
    data = pd.read_csv(filename)
    X_train,X_test,y_train,y_test = train_test_split(data[['ph', 'watertemp', 'light', 'humidity', 'airtemp']], data['dissolved_oxygen_mg_l'], test_size=0.05)
    model = StackingRegressor(estimators=[
        ('svr', NuSVR(kernel='poly', shrinking=False, C=2.5)), 
        ('rf', RandomForestRegressor(n_estimators=10,random_state=42, n_jobs=8)),
        ('bag', BaggingRegressor(n_jobs=5)),
        ('bst', GradientBoostingRegressor(learning_rate=0.1, loss='huber', max_depth=6, criterion='squared_error')),
        ('nn', MLPRegressor(hidden_layer_sizes=(2,3), activation='relu') )
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (model, stats(y_pred, y_test))
        



