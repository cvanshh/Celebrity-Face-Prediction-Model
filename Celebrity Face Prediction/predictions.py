import numpy as np
import pickle
from collections import Counter, defaultdict
from deepface import DeepFace

# Load embeddings database
with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Prediction function
def predict(img_path, top_k=3):

    # Get embedding of input image
    target = DeepFace.represent(
        img_path=img_path,
        model_name='Facenet',
        enforce_detection=False,
        detector_backend='opencv'
    )[0]['embedding']

    target = np.array(target)

    # Compare with database
    scores = []
    for entry in db:
        emb = np.array(entry["embedding"])
        score = cosine_similarity(target, emb)
        scores.append((entry["name"], score))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    # Top matches
    top_k_results = scores[:top_k]

    # Best match
    best_name, best_score = scores[0]

    # Confidence levels
    if best_score > 0.75:
        confidence = "High"
    elif best_score > 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    return best_name, best_score, top_k_results, confidence


# Test
name, score, top_k, confidence = predict("s3.png")

print("Prediction:", name)
print("Similarity Score:", round(score, 3))
print("Confidence:", confidence)

print("\nTop Matches:")
for n, s in top_k:
    print(f"{n} — {round(s,3)}")