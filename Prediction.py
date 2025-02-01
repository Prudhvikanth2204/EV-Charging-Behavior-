import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/content/station_data_dataverse12.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the SVR model with RBF kernel (default kernel)
svr = SVR(kernel='rbf', C=100, epsilon=0.1)

# Creating a pipeline that scales the data and then applies SVR
pipeline = make_pipeline(StandardScaler(), svr)

# Fitting the model
pipeline.fit(X, y)

# Making predictions
y_pred = pipeline.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Since SVR does not directly provide a classification output,
# accuracy is not applicable. For regression models, R² is a better metric.

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('Support Vector Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model
rf.fit(X, y)

# Making predictions
y_pred = rf.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('Random Forest Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the Decision Tree Regressor model
dt = DecisionTreeRegressor(random_state=42)

# Fitting the model
dt.fit(X, y)

# Making predictions
y_pred = dt.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('Decision Tree Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the Linear Regression model
lr = LinearRegression()

# Fitting the model
lr.fit(X, y)

# Making predictions
y_pred = lr.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('Linear Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values.reshape(-1, 1)

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape the data to fit LSTM input [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Creating the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X_scaled, y_scaled, epochs=150, batch_size=50, verbose=1)

# Making predictions
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('LSTM: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Fitting the model
xgb_model.fit(X, y)

# Making predictions
y_pred = xgb_model.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('XGBoost Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

# Load data
data = pd.read_csv('/content/EV_DATA.csv')

# Prepare the data
X = data['chargeTimeHrs'].values.reshape(-1, 1)
y = data['kwhTotal'].values

# Creating the LightGBM model
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Fitting the model
lgbm_model.fit(X, y)

# Making predictions
y_pred = lgbm_model.predict(X)

# Calculating R² score
r2 = r2_score(y, y_pred)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Output the metrics
print(f"R² score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual y values')
plt.plot(X, y_pred, color='blue', label='Predicted y values', linewidth=2)
plt.title('LightGBM Regression: Actual vs. Predicted')
plt.xlabel('Charge Time (Hrs)')
plt.ylabel('Total Energy Consumption (kWh)')
plt.legend()
plt.show()

