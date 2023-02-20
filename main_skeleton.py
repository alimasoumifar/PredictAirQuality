import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
air_quality_data = pd.read_csv('air_quality_data.csv')
weather_data = pd.read_csv('weather_data.csv')

# Merge data
merged_data = pd.merge(air_quality_data, weather_data, on='date')

# Split data into train and test sets
X = merged_data.drop(['date', 'air_quality_index'], axis=1)
y = merged_data['air_quality_index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# Train decision tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

# Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

# Save the best model
if lr_r2 > dt_r2 and lr_r2 > rf_r2:
    joblib.dump(lr_model, 'air_quality_predictor.pkl')
elif dt_r2 > lr_r2 and dt_r2 > rf_r2:
    joblib.dump(dt_model, 'air_quality_predictor.pkl')
else:
    joblib.dump(rf_model, 'air_quality_predictor.pkl')

# Deploy the model as an API
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temperature = data['temperature']
    humidity = data['humidity']
    wind_speed = data['wind_speed']
    model = joblib.load('air_quality_predictor.pkl')
    air_quality_index = model.predict([[temperature, humidity, wind_speed]])
    return jsonify({'air_quality_index': air_quality_index[0]})

if __name__ == '__main__':
    app.run(debug=True)
