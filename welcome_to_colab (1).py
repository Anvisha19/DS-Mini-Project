import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/content/sample_data/Electric_Vehicle_Population_Data.csv'
data = pd.read_csv(file_path)

# Display basic dataset info
print("Dataset Info:")
print(data.info())

# Display dataset head
print("\nDataset Head:")
print(data.head())

# Cleaning and preprocessing the dataset

# Ensure numeric columns are treated as numbers
data['Electric Range'] = pd.to_numeric(data['Electric Range'], errors='coerce')
data['Base MSRP'] = pd.to_numeric(data['Base MSRP'], errors='coerce')

# Drop rows with missing values in key columns
data = data.dropna(subset=['Electric Range', 'Model Year', 'Electric Vehicle Type'])

# Feature Engineering: Extracting key features
data['Vehicle Type'] = data['Electric Vehicle Type'].apply(
    lambda x: 'BEV' if 'Battery Electric Vehicle' in x else 'PHEV'
)

# Exploratory Data Analysis (EDA)

# Box Plot: Electric Range by Vehicle Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Vehicle Type', y='Electric Range', data=data)
plt.title('Electric Range by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Electric Range')
plt.show()

# Histogram: Electric Range Distribution
plt.figure(figsize=(10, 6))
plt.hist(data['Electric Range'], bins=30, edgecolor='k', color='skyblue')
plt.title('Distribution of Electric Range')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.show()

# Bar Plot: Count of Vehicles by Model Year
plt.figure(figsize=(10, 6))
sns.countplot(x='Model Year', data=data, order=sorted(data['Model Year'].unique()))
plt.title('Count of Vehicles by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Scatter Plot: Electric Range vs Base MSRP
plt.figure(figsize=(10, 6))
plt.scatter(data['Base MSRP'], data['Electric Range'], alpha=0.5, color='green')
plt.title('Electric Range vs Base MSRP')
plt.xlabel('Base MSRP ($)')
plt.ylabel('Electric Range (miles)')
plt.show()

# Preparing Data for Regression Analysis
# Define features (X) and target (y)
X = data[['Model Year', 'Base MSRP']]  # Features: Model Year and MSRP
y = data['Electric Range']  # Target: Electric Range

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")

# Scatter Plot: True vs Predicted Electric Range
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.title('True vs Predicted Electric Range')
plt.xlabel('True Electric Range (miles)')
plt.ylabel('Predicted Electric Range (miles)')
plt.show()
