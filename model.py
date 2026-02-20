import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv('real_estate123.csv')

# Preprocess the data
# Encode the 'Area' column as it is categorical
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])

# Define features and target variable
X = df[['Square_Feet', 'Bedrooms', 'Location', 'Age_of_Property']]  # Features
y = df['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('model1.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as 'model1.pkl'")

# Optionally, you can check the model's performance
accuracy = model.score(X_test, y_test)
print(f"Model R-squared on test data: {accuracy:.2f}")
