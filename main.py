from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained LSTM model
model = load_model("lstm_model.h5")

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("home.html")  # Assumes you have an index.html for your UI

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Assumes frontend sends JSON

        # Extract values in the order your model expects
        input_features = [
            data["time_step"],
            data["pressure"],
            data["temperature"],
            data["vibration"],
            data["corrosion"],
            data["flow_rate"],
            data["salinity"],
            data["age"],
            data["pressure_temp"],
            data["corrosion_flow"]
        ]

        # Reshape for LSTM input: (1, timesteps, features)
        input_array = np.array(input_features).reshape(1, 1, -1)

        # Predict using the model
        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map class to label
        label_map = {0: "Normal", 1: "Warning", 2: "Critical"}
        result = label_map.get(predicted_class, "Unknown")

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)