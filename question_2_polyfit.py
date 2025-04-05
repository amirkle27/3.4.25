import numpy as np
import matplotlib.pyplot as plt
x = np.array([2, 3, 4, 5, 6])
y = np.array([50, 70, 90, 100, 110])

m, b = np.polyfit(x, y, 1)

print(f"שיפוע (b1): {m:.2f}")
print(f"נקודת חיתוך עם ציר y (b0): {b:.2f}")



plt.figure(figsize=(10, 6))

# פיזור הנתונים המקוריים (הנקודות הכחולות)
plt.scatter(x, y, color='blue', label='Data points')

# קו הרגרסיה המחושב עם polyfit (קו אדום)
y_pred = m * x + b  # מחשבים את כל ערכי y לפי הקו
plt.plot(x, y_pred, color='red', label='Regression line')

# משוואת הקו
equation = f'y = {m:.2f}x + {b:.2f}'
plt.text(1.9, 121, equation, fontsize=12, color = 'green')

x_pred = 8
y_pred_8 = m * x_pred + b
print(f"אם תדבר {x_pred} שעות, הרווח הצפוי הוא: {y_pred_8:.2f}")

plt.scatter([x_pred], [y_pred_8], color='green', s=100, label=f'Prediction for 8 hours\nProfit = {y_pred_8}')
plt.legend(loc='upper left')  # כדי לעדכן גם את המקרא

# כותרות וצירים
plt.title('Calls Hours vs. Profit with np.polyfit')
plt.xlabel('Calls Hours')
plt.ylabel('Profit')
plt.grid(True)
plt.legend(loc = 'upper left')

plt.show()