import os
from utils import get_embedding, load_centroids
import numpy as np
from models import get_embedding_model
from config import THRESHOLDS

model, device = get_embedding_model()

def intra_class_test(person: str = "Fabris", max_images=30, mean_embeddings = [], mean_labels = []):
    try:
        threshold = THRESHOLDS[model.name][person]
    except KeyError:
        threshold = 0.5  # default threshold if not found in config

    if len(mean_embeddings) == 0 or len(mean_labels) == 0:
        mean_embeddings, mean_labels = load_centroids(model)
    
    person_idx = np.where(mean_labels == person)[0]
    if len(person_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return
    
    centroid = mean_embeddings[person_idx][0]
    
    person_path = os.path.join("dataset/train", person)
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    similarities = []
    correct = 0
    tested = 0

    print(f"Testiram {person} | Threshold: {threshold} | Centroid norm: {np.linalg.norm(centroid):.4f}\n")

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb = get_embedding(img_path)   # koristi utils verziju
        
        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue
            
        sim = float(np.dot(centroid, emb))
        similarities.append(sim)
        
        label = person if sim > threshold else "Unknown"
        is_correct = (label == person)
        
        if is_correct:
            correct += 1
        tested += 1
        
        print(f"{img_name:35} → sim: {sim:.4f} → {label}")

    if tested == 0:
        print("Nema validnih embeddinga!")
        return

    accuracy = correct / tested * 100
    print("\n" + "="*60)
    print(f"REZULTAT ZA {person.upper()}")
    print(f"Točnost: {accuracy:.1f}% ({correct}/{tested})")
    print(f"Prosječna sličnost: {np.mean(similarities):.4f}")
    print(f"Max sličnost: {max(similarities):.4f}")
    print(f"Min sličnost: {min(similarities):.4f}")
    print(f"20. percentil: {np.percentile(similarities, 20):.4f}")
    print("="*60)
    
    return similarities

def inter_class_test(person1: str, person2: str, max_images=30, mean_embeddings = [], mean_labels = []):
    
    if len(mean_embeddings) == 0 or len(mean_labels) == 0:
        mean_embeddings, mean_labels = load_centroids(model)

    person1_idx = np.where(mean_labels == person1)[0]
    person2_idx = np.where(mean_labels == person2)[0]
    if len(person1_idx) == 0 or len(person2_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return
    
    centroid = mean_embeddings[person2_idx][0]
    #print("Centroid ", centroid)
    
    person_path = os.path.join("dataset/train", person1)
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    similarities = []
    similar = 0
    tested = 0

    print(f"Testiram {person1} | Threshold: {THRESHOLDS[model.name][person1]} | Centroid norm: {np.linalg.norm(centroid):.4f}\n s Osobom: {person2}")

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb = get_embedding(img_path)   # koristi utils verziju
        
        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue
        #print("Embbedding ", emb)
        sim = float(np.dot(centroid, emb))
        similarities.append(sim)
        
        label = "Not similar" if sim < THRESHOLDS[model.name][person1] else "Similar"
        is_similar = (label == "Similar")
        
        if is_similar:
            similar += 1
        tested += 1
        
        print(f"{img_name:35} → sim: {sim:.4f} → {label}")

    if tested == 0:
        print("Nema validnih embeddinga!")
        return

    accuracy = similar / tested * 100
    print("\n" + "="*60)
    print(f"REZULTAT ZA {person1.upper()} U USPOREDBI S {person2.upper()}")
    print(f"Sličnost: {accuracy:.1f}% ({similar}/{tested})")
    print(f"Prosječna sličnost: {np.mean(similarities):.4f}")
    print(f"Max sličnost: {max(similarities):.4f}")
    print(f"Min sličnost: {min(similarities):.4f}")
    print(f"90. percentil: {np.percentile(similarities, 90):.4f}")
    print("="*60)
    
    return similarities

if __name__ == "__main__":
    for person in os.listdir(r"dataset\train"):
        similarities = inter_class_test("Mathias", person, max_images=30)
    #     if(person != "Fabris"):
        