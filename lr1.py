import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('eda_data.csv')

# Data Preprocessing

# Drop unnecessary columns that do not impact salary
data = data[['Job Title', 'Rating', 'Location', 'Size', 'Type of ownership', 'Industry', 'Sector',
             'Revenue', 'avg_salary', 'python_yn', 'excel', 'job_simp', 'seniority']]

# Drop rows with missing values in relevant columns
data = data.dropna(subset=['avg_salary'])

# Define features (X) and target (y)
X = data.drop(columns=['avg_salary'])
y = data['avg_salary']

# Numerical and categorical features
numeric_features = ['Rating']
categorical_features = ['Job Title', 'Location', 'Size', 'Type of ownership', 'Industry', 'Sector',
                        'Revenue', 'python_yn', 'excel', 'job_simp', 'seniority']

# Preprocessing: One-hot encode categorical features and scale numerical features
# Update the OneHotEncoder to ignore unknown categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# Create a pipeline with preprocessing and the Linear Regression model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print(pipeline.score(X_test,y_test))