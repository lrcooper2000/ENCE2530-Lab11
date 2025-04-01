## step 1: Data Exploration and Preprocessing:

import pandas as pd

# Load the dataset
df = pd.read_csv("lab_11_bridge_data.csv")

# Display basic info
print(df.info())
print(df.head())

print(df.isnull().sum())  # Check for missing values


df = pd.get_dummies(df, columns=["Material"], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns_to_scale = ["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", "Condition_Rating"]
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


##Learning Code

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

y_pred = model.predict(X_test)
y_pred = (y_pred * y_std) + y_mean


# Save the scaler for later use in deployment
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Bridge_ID", "Max_Load_Tons"])  # Features
y = df["Max_Load_Tons"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# Define the model
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Show model summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

import numpy as np

# Convert to NumPy arrays with float32 dtype
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

# Check the dtype
print(X_train.dtype, y_train.dtype)  # Should print float32

# Train the model
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=64, 
                    callbacks=[early_stop])

import matplotlib.pyplot as plt

# Plot loss over epochs
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.2f} tons")

model.save("tf_bridge_model.h5")

