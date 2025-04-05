import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Our data
advertising_investment = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)  # Study hours
sales_growth = np.array([25, 30, 40, 45, 50, 60, 65, 70, 80])  # Exam scores

# Create regression model
model = LinearRegression()
model.fit(advertising_investment, sales_growth)

# Print results
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")

# Calculate equation
equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
print(f"Line equation: {equation}")

# Predict hours needed to get score of 100
# score_to_predict = 100
# hours_needed = (score_to_predict - model.intercept_) / model.coef_[0]
# print(f"To get a score of 100, approximately {hours_needed:.2f} hours of study are needed")

# Create the graph
plt.figure(figsize=(10, 6))
plt.scatter(advertising_investment, sales_growth, color='blue', label='Data points')
plt.plot(advertising_investment, model.predict(advertising_investment), color='red', label='Regression line')

# Add prediction point
# plt.scatter(advertising_investment, sales_growth, color='green', s=100, label='Data Points')

# Add labels in English
plt.title('Linear Regression - Advertising Investment vs. Sales Growth')
plt.xlabel('Advertising Investment (in 1000 ILS)')
plt.ylabel('Sales Growth (in 1000 ILS)')
plt.grid(True)
plt.legend()

m = model.coef_[0]
b = model.intercept_
equation = f'y = {m:.2f}x + {b:.2f}'
#plt.text(10, 75, equation, fontsize=12, color='green')

# Display equation on the graph
plt.text(9, 72, equation, fontsize=12, color = 'green')

plt.show()