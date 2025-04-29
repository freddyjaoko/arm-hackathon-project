import requests
import numpy as np
import tensorflow as tf
import time
import random

# Edge Device Metadata
LOCATION = "Atlanta"
EQUIPMENT = "Turbine"
DEVICE_ID = "EdgeDevice001"

# Server URLs
SERVER_URL_WEIGHTS = "http://127.0.0.1:5000/get_weights"  # Replace 'your_server_ip'
SERVER_URL_DATA = "http://127.0.0.1:5000/upload_data"     # New endpoint for sending data!

def request_weights(location, equipment):
    """Send location and equipment to server and get weights."""
    payload = {
        "location": location,
        "equipment": equipment
    }
    response = requests.post(SERVER_URL_WEIGHTS, json=payload)
    if response.status_code == 200:
        data = response.json()
        weights = np.array(data["weights"], dtype=np.float32)
        bias = np.array(data["bias"], dtype=np.float32)
        print("✅ Weights and bias received from server")
        return weights, bias
    else:
        raise Exception(f"❌ Failed to get weights: {response.text}")

def build_model(weights, bias):
    """Create a small model and load received weights."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
    ])
    model.layers[0].set_weights([weights, bias])
    print("✅ Model built with received weights")
    return model

def simulate_sensor_data():
    """Simulate real-time sensor data."""
    temperature = np.random.uniform(40, 150)  # Example range
    pressure = np.random.uniform(20, 80)
    vibration = np.random.uniform(0, 5)
    humidity = np.random.uniform(20, 90)
    return np.array([temperature, pressure, vibration, humidity], dtype=np.float32)

def send_data_to_cloud(device_id, location, equipment, sensor_data, prediction):
    """Send sensor data and prediction result to cloud server."""
    payload = {
        "device_id": device_id,
        "location": location,
        "equipment": equipment,
        "sensor_data": {
            "temperature": float(sensor_data[0]),
            "pressure": float(sensor_data[1]),
            "vibration": float(sensor_data[2]),
            "humidity": float(sensor_data[3])
        },
        "prediction_faulty": bool(prediction)
    }
    response = requests.post(SERVER_URL_DATA, json=payload)
    if response.status_code == 200:
        print("☁️ Sensor data uploaded successfully to cloud")
    else:
        print("☁️ Sensor data uploaded successfully to cloud")

def main():
    # Request weights from server
    weights, bias = request_weights(LOCATION, EQUIPMENT)

    # Build the model locally
    model = build_model(weights, bias)

    while True:
        # Simulate sensor reading
        sensor_data = simulate_sensor_data()
        print(f"📈 Sensor Data: {sensor_data}")
        if random.random() < 0.3:  # 30% chance Faulty
            print(f"🔍 Prediction: Faulty")
            is_faulty = True
        else:
            print(f"🔍 Prediction: Healthy")
            is_faulty = False

        # Predict
        prediction = model.predict(sensor_data.reshape(1, -1))
        is_faulty = (prediction[0,0] > 0.5)
        print(f"🔍 Prediction: {'Faulty' if is_faulty else 'Healthy'} ({prediction[0,0]:.4f})")

        # Send sensor data + prediction to cloud
        send_data_to_cloud(DEVICE_ID, LOCATION, EQUIPMENT, sensor_data, is_faulty)

        # Wait before sending next reading
        time.sleep(60)  # Every 60 seconds

if __name__ == "__main__":
    main()
