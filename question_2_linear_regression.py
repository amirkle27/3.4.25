import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Our data
calls_hours = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)  # Study hours
profit = np.array([50, 70, 90, 100, 110])  # Exam scores

# Create regression model
model = LinearRegression()
model.fit(calls_hours, profit)

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
plt.scatter(calls_hours, profit, color='blue', label='Data points')
plt.plot(calls_hours, model.predict(calls_hours), color='red', label='Regression line')

# Add prediction point
# plt.scatter(advertising_investment, sales_growth, color='green', s=100, label='Data Points')

# Add labels in English
plt.title('Linear Regression - Calls Hours vs. Profit')
plt.xlabel('Calls Hours')
plt.ylabel('Profit')
plt.grid(True)
plt.legend(loc = 'upper left')

m = model.coef_[0]
b = model.intercept_
equation = f'y = {m:.2f}x + {b:.2f}'

x_pred = 8
y_pred = m * x_pred + b
print(f"רווח צפוי אחרי {x_pred} שעות שיחה: {y_pred:.2f}")

#plt.text(10, 75, equation, fontsize=12, color='green')

# Display equation on the graph
plt.text(1.9, 133, equation, fontsize=12, color = 'green')

plt.scatter([x_pred], [y_pred], color='green', s=100, label=f'Prediction for 8 hours\nProfit = {y_pred}')


plt.show()

