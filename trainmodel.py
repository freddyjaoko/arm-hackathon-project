import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

MODEL_FOLDER = "models"

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

train_and_save_global_model("equipment_anomaly_data.csv")