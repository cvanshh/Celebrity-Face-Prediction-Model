import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset_path = "faces"

from deepface import DeepFace
import os
import pickle

dataset_path = "faces"
embeddings = []

total = 0
skipped = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        total += 1

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name='Facenet',
                enforce_detection=False
            )

            embeddings.append({
                "name": person,
                "embedding": rep[0]['embedding']
            })

        except Exception as e:
            skipped += 1
            print("Skipped:", img_path)

# Save embeddings
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Done")
print("Total images:", total)
print("Embeddings created:", len(embeddings))
print("Skipped:", skipped)

