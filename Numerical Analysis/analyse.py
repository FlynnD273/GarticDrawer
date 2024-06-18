import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the CSV file
file_path = 'rounds.csv'
data = pd.read_csv(file_path)

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Prepare the data
X = data[['Total', 'Top']]
y = data['Rounds']

# Transform the features to polynomial features
poly = PolynomialFeatures(degree=2)  # Change the degree as needed
X_poly = poly.fit_transform(X)

# Initialize and fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Evaluate the model
r2 = r2_score(y, y_pred)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the results
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("R-squared:", r2)

# Print the equation
feature_names = poly.get_feature_names_out(['Total', 'Top'])
equation = f"Rounds = {intercept:.4f} "
for coef, name in zip(coefficients, feature_names):
    if name != '1':
        equation += f"+ {coef:.4f} * {name} "

print(equation)

