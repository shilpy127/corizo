import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('/content/1729258-1613615-Stock_Price_data_set_(1).csv')
X = df[['Open']]
X
Y = df['Close']
Y
df.shape
df.head(5)
df.info()
df.describe
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)
M = LinearRegression()
M.fit(X_train, Y_train)
Y_pred=M.predict(X_test)
plt.scatter(X_test, Y_test, color='red', label='Actual Prices')
plt.plot(X_test, Y_pred, color='black', linewidth=3, label='Predicted Prices')
plt.xlabel('opening price')
plt.ylabel('closing price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
