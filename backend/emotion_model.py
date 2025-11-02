from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

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
    """Normalize the transition probabilities."""
    normalized = {}
    for e1 in labels:
        total = sum(transition_matrix[e1].values()) or 1
        normalized[e1] = {e2: round(transition_matrix[e1][e2]/total, 2)
                          for e2 in labels}
    return normalized
