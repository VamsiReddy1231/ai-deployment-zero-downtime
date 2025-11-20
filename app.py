import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = "model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Metric for monitoring
prediction_counter = Counter("predictions_total", "Total predictions served")

@app.route("/healthz")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("input")

    if not data:
        return jsonify({"error": "Input missing"}), 400

    prediction_counter.inc()
    prediction = model.predict([data]).tolist()

    return jsonify({"prediction": prediction}), 200

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
