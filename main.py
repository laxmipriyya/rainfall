import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create a Sample Dataset
data = {
    'Temperature': [30, 32, 35, 40, 42, 45, 50, 55, 60, 65],
    'Humidity': [70, 75, 80, 85, 87, 90, 92, 95, 97, 99],
    'Pressure': [1010, 1008, 1005, 1003, 1000, 998, 995, 993, 990, 985],
    'Rainfall': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}

# Step 2: Load Data into a DataFrame
df = pd.DataFrame(data)
print("Sample Data:")
print(df.head())

# Step 3: Define Features and Target
X = df[['Temperature', 'Humidity', 'Pressure']]
y = df['Rainfall']

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Predict Rainfall for New Data
new_data = np.array([[33, 78, 1007]])  # Example: Temperature=33, Humidity=78, Pressure=1007
predicted_rainfall = model.predict(new_data)
print("Predicted Rainfall:", predicted_rainfall[0])
