# import nedded libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create the Dataset

Data = {
    'first_hour_likes': [120, 340, 89, 510, 230, 670, 45, 390, 150, 720,
                         280, 95, 440, 180, 560, 75, 320, 480, 210, 630,
                         110, 350, 85, 410, 260, 590, 140, 470, 55, 380],
    
    'total_views': [8500, 22000, 5200, 41000, 15000, 53000, 3200, 28000, 10500, 58000,
                    18500, 6800, 32000, 12000, 44000, 4500, 21000, 36000, 14000, 49000,
                    7800, 23500, 5000, 30000, 17000, 46000, 9500, 35000, 3800, 27000]
}


df = pd.DataFrame(Data)

# Prepare Data for Training

x = df[['first_hour_likes']]
y = df['total_views']

# Split into Training and Testing Sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and Train the Model, Make Predictions

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Make Prediction for new data

new_likes = 750
predicted_views = model.predict([[new_likes]])

# Output the Prediction

print(f"If raj get {new_likes} first hour likes, how many Total Views can he get?")
print(f"Predicted Total Views by using linear regression: {predicted_views[0]:.0f}")

# Visualize the Results

plt.figure(figsize=(10,6))
plt.scatter(df['first_hour_likes'], df['total_views'], color='blue', s=100, label='Actual Views')

plt.plot(df['first_hour_likes'], model.predict(df[['first_hour_likes']]), color='red', linewidth=2, label='Regression Line')
plt.scatter(new_likes, predicted_views, color='green', s=200, marker='*', label='Predicted Views')

plt.xlabel('First Hour Likes', fontsize=12)
plt.ylabel('Total Views', fontsize=12)

plt.title('First Hour Likes vs Total Views Prediction', fontsize=14)

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('likes_total_views_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'likes_total_views_prediction.png'")

