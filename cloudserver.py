import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

"""

"""
def train_and_save_global_model(csv_path):
    """Train one model on all data and save it."""

    df = pd.read_csv(csv_path)

    expected_columns = ["temperature", "pressure", "vibration", "humidity", "faulty", "equipment", "location"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"CSV missing columns. Expected: {expected_columns}")

    X = df[["temperature", "pressure", "vibration", "humidity"]].values
    y = df["faulty"].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Save full model
    model.save(os.path.join(MODEL_FOLDER, "global_model.h5"))

    # Save weights separately (optional)
    weights, bias = model.layers[0].get_weights()
    np.save(os.path.join(MODEL_FOLDER, "weights.npy"), weights)
    np.save(os.path.join(MODEL_FOLDER, "bias.npy"), bias)

    print("✅ Model trained and weights saved.")


@app.route('/get_weights', methods=['POST'])
def get_weights_for_device():
    """Receive location and equipment and return weights + bias."""

    data = request.json
    location = data.get('location')
    equipment = data.get('equipment')

    # For now we have only global weights
    weights_path = os.path.join(MODEL_FOLDER, "weights.npy")
    bias_path = os.path.join(MODEL_FOLDER, "bias.npy")

    if not os.path.exists(weights_path) or not os.path.exists(bias_path):
        return jsonify({"error": "Weights not found. Train model first."}), 404

    # Load weights
    weights = np.load(weights_path)
    bias = np.load(bias_path)

    # Prepare response
    response = {
        "location": location,
        "equipment": equipment,
        "weights": weights.tolist(),
        "bias": bias.tolist()
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
