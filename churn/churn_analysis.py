import pandas as pd

# Load the Excel file
file_path = "your_file.xlsx"  # Replace with your actual file name
df = pd.read_excel(file_path)

# Keep only the selected columns
selected_columns = ["Status", "Gender", "Country", "Institution Name", "Major", "Opportunity Type"]
df = df[selected_columns]

# Check for missing values
print("Missing values before handling:\n", df.isnull().sum())

# Fill or drop missing values (Choose one)
df = df.dropna()  # Remove rows with missing values
# OR
# df = df.fillna("Unknown")  # Replace missing values with 'Unknown'

# Display the first few rows
print(df.head())

# Convert Categorical Data into Numbers

from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding
encoder = LabelEncoder()
for col in ["Status", "Gender", "Country", "Institution Name", "Major", "Opportunity Type"]:
    df[col] = encoder.fit_transform(df[col])

# Display the transformed dataset
print(df.head())


# Define Churn (Target Variable)
# 1 = Dropped Out
# 0 = Completed

# Define the target variable
df["Status"] = df["Status"].apply(lambda x: 1 if x == 1 else 0)  # Adjust based on encoding

# Display dataset after defining churn
print(df.head())

# Split Data into Training & Testing Sets
#Split the dataset into features (X) and target (y).
#Divide it into 80% training and 20% testing data.

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=["Status"])  # Features
y = df["Status"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# Train a Machine Learning Model
# Use a classification algorithm to train the model.

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Evaluate the Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Interpret & Use Insights
# Identify important factors that influence churn.

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance
feature_importance = model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance")
plt.show()

