from flask import Flask, request, jsonify, render_template
from emotion_model import predict_emotion, get_emotion_clusters, get_transition_matrix
import random
import json

app = Flask(__name__, static_folder="static", template_folder="templates")

# Context-aware adaptive replies
RESPONSES = json.load(open("../data/emotion_responses.json", "r"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "Message field missing"}), 400

    emotion, emb = predict_emotion(message)
    response_list = RESPONSES.get(emotion, RESPONSES["neutral"])
    bot_reply = random.choice(response_list)

    # Generate clustering visualization after 5+ interactions
    clusters = get_emotion_clusters()
    transition_probs = get_transition_matrix()

    return jsonify({
        "predicted_emotion": emotion,
        "bot_reply": bot_reply,
        "transition_probs": transition_probs
    })
if __name__ == "__main__":
    app.run(debug=True)
    '''
def get_transition_matrix():
    """Returns normalized transition probabilities."""
    normalized_matrix = {}
    for e1, transitions in transition_matrix.items():
        total = sum(transitions.values())
        if total > 0:
            normalized_matrix[e1] = {e2: count / total for e2, count in transitions.items()}
        else:
            normalized_matrix[e1] = {e2: 0 for e2 in transitions}
    return normalized_matrix
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os       
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import joblib
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', '
sadness', 'surprise']
# Emotion transition memory (Markov-like)
emotion_memory = []
transition_matrix = {e1: {e2: 0 for e2 in labels} for e1 in labels}
def transformer_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = labels[torch.argmax(probs)]
    return pred_label, probs.numpy().flatten()
def sentiment_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3: return "positive"
    elif polarity < -0.3: return "negative"
    return "neutral"
def update_transition(prev, current):
    if prev and current:
        transition_matrix[prev][current] += 1
def predict_emotion(text):
    transformer_label, emb = transformer_emotion(text)
    sentiment = sentiment_tone(text)
    last_emotion = emotion_memory[-1] if emotion_memory else None
    emotion_memory.append(transformer_label)
    update_transition(last_emotion, transformer_label)

    if transformer_label == "neutral" and sentiment == "negative":
        if any(w in text.lower() for w in ["tired", "lonely", "work", "stress"]):
            transformer_label = "sadness"

    return transformer_label, emb
def get_emotion_clusters():
    """Clusters the emotion embeddings seen so far using KMeans."""
    if len(emotion_memory) < 5:
        return None

    X = []
    for text in emotion_memory:
        _, emb = transformer_emotion(text)
        X.append(emb)
    X = np.array(X)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=k
means.labels_, cmap='viridis')
    plt.title("Emotion Embedding Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    cluster_path = "static/emotion_clusters.png"
    plt.savefig(cluster_path)
    plt.close()
    return cluster_path
def get_transition_matrix():
    """Returns normalized transition probabilities."""
    normalized_matrix = {}
    for e1, transitions in transition_matrix.items():
        total = sum(transitions.values())
        if total > 0:
            normalized_matrix[e1] = {e2: count / total for e2, count in transitions.items()}
        else:
            normalized_matrix[e1] = {e2: 0 for e2 in transitions}
    return normalized_matrix
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = labels[torch.argmax(probs)]
    return pred_label, probs.numpy().flatten()
def sentiment_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3: return "positive"
    elif polarity < -0.3: return "negative"
    return "neutral"
def update_transition(prev, current):
    if prev and current:
        transition_matrix[prev][current] += 1
def predict_emotion(text):
    transformer_label, emb = transformer_emotion(text)
    sentiment = sentiment_tone(text)
    last_emotion = emotion_memory[-1] if emotion_memory else None
    emotion_memory.append(transformer_label)
    update_transition(last_emotion, transformer_label)

    if transformer_label == "neutral" and sentiment == "negative":
        if any(w in text.lower() for w in ["tired", "lonely", "work", "stress"]):
            transformer_label = "sadness"

    return transformer_label, emb
def get_emotion_clusters():
    """Clusters the emotion embeddings seen so far using KMeans."""
    if len(emotion_memory) < 5:
        return None

    X = []
    for text in emotion_memory:
        _, emb = transformer_emotion(text)
        X.append(emb)

    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(6,4))
    plt.scatter(reduced[:,0], reduced[:,1], c=y_pred, cmap="viridis")
    plt.title("Emotion Cluster Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("../data/emotion_clusters.png")
    plt.close()

    return y_pred
def get_transition_matrix():
    """Returns normalized transition probabilities."""
    normalized_matrix = {}
    for e1, transitions in transition_matrix.items():
        total = sum(transitions.values())
        if total > 0:
            normalized_matrix[e1] = {e2: count / total for e2, count in transitions.items()}
        else:
            normalized_matrix[e1] = {e2: 0 for e2 in transitions}
    return normalized_matrix
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = labels[torch.argmax(probs)]
    return pred_label, probs.numpy().flatten()
def sentiment_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3: return "positive"
    elif polarity < -0.3: return "negative"
    return "neutral"
def update_transition(prev, current):
    if prev and current:
        transition_matrix[prev][current] += 1
def predict_emotion(text):
    transformer_label, emb = transformer_emotion(text)
    sentiment = sentiment_tone(text)
    last_emotion = emotion_memory[-1] if emotion_memory else None
    emotion_memory.append(transformer_label)
    update_transition(last_emotion, transformer_label)

    if transformer_label == "neutral" and sentiment == "negative":
        if any(w in text.lower() for w in ["tired", "lonely", "work", "stress"]):
            transformer_label = "sadness"

    return transformer_label, emb
def get_emotion_clusters(): 
    """Clusters the emotion embeddings seen so far using KMeans."""
    if len(emotion_memory) < 5:
        return None

    X = []
    for text in emotion_memory:
        _, emb = transformer_emotion(text)
        X.append(emb)

    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(6,4))
    plt.scatter(reduced[:,0], reduced[:,1], c=y_pred, cmap="viridis")
    plt.title("Emotion Cluster Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("../data/emotion_clusters.png")
    plt.close()

    return y_pred
'''