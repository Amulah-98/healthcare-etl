import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the CSV file
file_path = '..\healthcare\Heart_Disease_Prediction.csv'
heart_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
heart_data.head()

# Check for missing values
missing_values = heart_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values (if any) - here we drop rows with missing values
heart_data.dropna(inplace=True)

# Transform categorical variables
label_encoder = LabelEncoder()

# Convert 'Heart Disease' column to binary (0 for Absence, 1 for Presence)
heart_data['Heart Disease'] = label_encoder.fit_transform(heart_data['Heart Disease'])

# Convert other categorical columns
heart_data['Sex'] = label_encoder.fit_transform(heart_data['Sex'])
heart_data['Chest pain type'] = label_encoder.fit_transform(heart_data['Chest pain type'])
heart_data['FBS over 120'] = label_encoder.fit_transform(heart_data['FBS over 120'])
heart_data['EKG results'] = label_encoder.fit_transform(heart_data['EKG results'])
heart_data['Exercise angina'] = label_encoder.fit_transform(heart_data['Exercise angina'])
heart_data['Slope of ST'] = label_encoder.fit_transform(heart_data['Slope of ST'])
heart_data['Number of vessels fluro'] = label_encoder.fit_transform(heart_data['Number of vessels fluro'])
heart_data['Thallium'] = label_encoder.fit_transform(heart_data['Thallium'])

# Normalize numerical features
scaler = StandardScaler()

numerical_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
heart_data[numerical_features] = scaler.fit_transform(heart_data[numerical_features])

# Save the cleaned dataset
cleaned_file_path = '../healthcare/Cleaned_Heart_Disease_Prediction.csv'
heart_data.to_csv(cleaned_file_path, index=False)

print("Data cleaning and transformation complete. Cleaned dataset saved to:", cleaned_file_path)
