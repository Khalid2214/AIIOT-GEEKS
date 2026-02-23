import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2D array dataset

data = [
[500, 30],
[1000, 38],
[1500, 45],
[2000, 55],
[2500, 65],
[3000, 70],
[3500, 78],
[400, 90],
[4500, 100],
[4500, 110]
]

data = np.array(data)
#
#2. Split into X and y
#
X = data[:, 0].reshape(-1, 1) #column 0 = area
y = data[:, 1]
#column 1 = price
#
#3. Train-test split
#
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
#
#4. Train model
#
model = LinearRegression()
model.fit(X_train, y_train)
#
#5. Predictions
#
y_pred = model.predict(X_test)
#
#6. Evaluation
#
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("\nActual vs Predicted:")


for a, p in zip(y_test, y_pred):
    
    print(f"Actual: {a}, Predicted: {p:.2f}")


print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
#
#7. Plot
#
plt.scatter(X, y, label="Data Points")
plt.plot(X, model.predict(X), label="Fit Line")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("LinearRegression (2D Array Input)")
plt.legend()
plt.show()