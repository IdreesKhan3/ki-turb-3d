import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Data Generation (for demonstration purposes) ---
# In a real scenario, you would load your actual strain rate, vorticity, and stress data.

def generate_dummy_data(num_samples=1000):
    # Features:
    # Strain rate components (S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) - 6 components
    # Vorticity vector components (omega_x, omega_y, omega_z) - 3 components
    # Total features = 9
    X_strain_rate = np.random.rand(num_samples, 6) * 10 - 5 # Random values between -5 and 5
    X_vorticity = np.random.rand(num_samples, 3) * 2 - 1   # Random values between -1 and 1
    X = np.hstack((X_strain_rate, X_vorticity))

    # Targets:
    # Stress tensor components (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz) - 6 components
    # Let's create a simple non-linear relationship for demonstration
    y_xx = 2 * X[:, 0] + 0.5 * X[:, 6]**2 + np.random.randn(num_samples) * 0.1
    y_yy = 1.5 * X[:, 1] - 0.3 * X[:, 7] + np.random.randn(num_samples) * 0.1
    y_zz = 1.8 * X[:, 2] + 0.2 * X[:, 8]**3 + np.random.randn(num_samples) * 0.1
    y_xy = 0.7 * X[:, 3] + X[:, 0] * X[:, 6] + np.random.randn(num_samples) * 0.1
    y_xz = 0.9 * X[:, 4] - 0.5 * X[:, 7] * X[:, 8] + np.random.randn(num_samples) * 0.1
    y_yz = 1.1 * X[:, 5] + 0.2 * X[:, 6] * X[:, 7] + np.random.randn(num_samples) * 0.1

    y = np.vstack((y_xx, y_yy, y_zz, y_xy, y_xz, y_yz)).T

    return X, y

# Generate data
X, y = generate_dummy_data(num_samples=5000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for MLP performance)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# It's also good practice to scale targets if their ranges vary widely,
# but for stress components, they might be in similar ranges. Let's skip for now.
# scaler_y = StandardScaler()
# y_train_scaled = scaler_y.fit_transform(y_train)
# y_test_scaled = scaler_y.transform(y_test)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train shape: {y_train.shape}")

# --- 2. Define the MLP Model Architecture ---
def build_mlp_model(input_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_dim) # Linear activation for regression output
    ])
    return model

# Input and output dimensions
input_dim = X_train_scaled.shape[1]  # 9 features
output_dim = y_train.shape[1]      # 6 stress components

# Build the model
model = build_mlp_model(input_dim, output_dim)

# Display model summary
model.summary()

# --- 3. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression tasks
    metrics=['mae'] # Mean Absolute Error for easier interpretability
)

# --- 4. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50, # You might need more epochs for real data
    batch_size=32,
    validation_split=0.1, # Use a portion of training data for validation
    verbose=1
)

print("\nModel training finished.")

# --- 5. Evaluate the Model ---
print("\nEvaluating model on test data...")
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# --- 6. Make Predictions (Example) ---
print("\nMaking predictions on a sample...")
# Take a few samples from the test set
sample_features_scaled = X_test_scaled[:5]
sample_true_stress = y_test[:5]

# Predict
sample_predictions = model.predict(sample_features_scaled)

print("\nSample Predictions vs. True Values:")
for i in range(len(sample_predictions)):
    print(f"--- Sample {i+1} ---")
    print(f"Predicted Stress: {sample_predictions[i]}")
    print(f"True Stress:      {sample_true_stress[i]}")
    print(f"Difference:       {sample_predictions[i] - sample_true_stress[i]}\n")

# --- 7. Save the Model (Optional) ---
# model.save('stress_prediction_mlp.h5')
# print("Model saved as 'stress_prediction_mlp.h5'")

# To load later:
# loaded_model = keras.models.load_model('stress_prediction_mlp.h5')
