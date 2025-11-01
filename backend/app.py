from flask import Flask, request, jsonify, render_template
import joblib
import nltk
import os

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model and vectorizer
MODEL_PATH = "../models/emotion_model.pkl"
VECTORIZER_PATH = "../models/vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print("⚠️ Error loading model/vectorizer:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message field is missing"}), 400

    try:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        return jsonify({"predicted_emotion": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
