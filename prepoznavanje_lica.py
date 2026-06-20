import time

from Embedding_model import get_embedding_model, get_transform
from PIL import Image
import os
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_embedding
from imutils.video import FPS, FileVideoStream
from detectors import FaceDetector
from tqdm import tqdm
import random
from metrics import RecognitionStats


model, device = get_embedding_model()
transform = get_transform()

# === DETEKTOR (promijeni po potrebi) ===
detector = FaceDetector(method="mtcnn")   # "haar" za RPi, "mtcnn" za jači stroj

mean_embs = np.load("centroid_znacajke.npy")
mean_labels = np.load("oznake.npy")


def recognize(emb_new, thr=0.85):
    sim = np.dot(mean_embs, emb_new)
    idx = np.argmax(sim)
    if sim[idx] > thr:
        return mean_labels[idx], sim[idx]
    else:
        return "Unknown", sim[idx]


def facial_recognition(thr=0.65):
    #cap = cv2.VideoCapture(0)  # Koristi kameru
    #cap = cv2.VideoCapture("video_Matej_20260518_165925.mp4")  # za testiranje videa
    cap = cv2.VideoCapture("video_Mathias_20260518_163440.mp4")  # za testiranje videa
    
    #v_len = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    fps_text = ''
    text = ''
    embedding = 0
    recognition = 0
    avg_detection = []
    avg_embedding = []
    stats = RecognitionStats()
    start_time = time.perf_counter()
    fps_imutils = FPS().start()
    fps = 0

    if not cap.isOpened():
        print("Nemogu otvoriti kameru")
        return

    while True:
    #for _ in range(v_len):
        ret, frame = cap.read()
        # frame = cap.read()
        frame_count += 1
        if frame is None:
            print("Nisam u mogućnosti pročitati frame")
            break

        # --- Detekcija lica ---
        before_detection = time.perf_counter()
        if(detector.method == "mtcnn"):
            boxes, probs = detector.detect(frame)          # vraća listu [x1, y1, x2, y2]
        elif(detector.method == "haar"):
            boxes = detector.detect(frame)
        after_detection = time.perf_counter()
        detection = after_detection - before_detection

        #print("[DEBUG] Vjerojatnost da je detektirano lice je: ", probs)
        # --- Obrada svakog detektiranog lica ---
        for box in boxes:
            x1, y1, x2, y2 = box

            # sigurnosna rezanja
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                before_embedding = time.perf_counter()
                emb_new = get_embedding_from_crop(face_crop)   # ili get_embedding(face_crop)
                after_embedding = time.perf_counter()

                before_recognition = time.perf_counter()
                label, dist = recognize(emb_new, thr=thr)
                stats.update(label)
                after_recognition = time.perf_counter()

                text = f"{label} ({dist:.2f})"
                embedding = after_embedding - before_embedding
                recognition = after_recognition - before_recognition

            except Exception as e:
                text = "error"
                embedding = 0
                recognition = 0

            # Crtanje
            before_draw = time.perf_counter()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            after_draw = time.perf_counter()
            draw = after_draw - before_draw

            avg_detection.append(detection)
            avg_embedding.append(embedding)

        # --- FPS i logiranje ---
        fps_imutils.update()
        fps_text = f"FPS: {fps:.1f}"

        if frame_count % 30 == 0 and frame_count > 0:
            now = time.perf_counter()
            elapsed = now - start_time
            fps = 30 / elapsed
            print(f"Vrijeme detekcije: {detection*1000:.1f} ms | "
                  f"Embedding: {embedding*1000:.1f} ms | "
                  f"Prepoznavanje: {recognition*1000:.1f} ms | "
                  f"Sveukupno: {elapsed*1000/30:.1f} ms")
            start_time = now

        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Facial recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Prosječni FPS: {:.1f}".format(fps_imutils.fps()))
    if avg_detection:
        print(f"Prosječno vrijeme detekcije: {np.mean(avg_detection)*1000:.1f} ms")
    if avg_embedding:
        print(f"Prosječno vrijeme embeddinga: {np.mean(avg_embedding)*1000:.1f} ms")
    stats.print_report()

def get_embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Ostaje isti - koristi se u videu"""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)        
    emb_np = emb.cpu().numpy()[0]
    return emb_np / np.linalg.norm(emb_np)


# === Ostale test funkcije (ostavljene, ali mtcnn uklonjen) ===
def validation_test():
    for dataset_path in ["dataset/random"]:
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for i, img_name in enumerate(os.listdir(class_path)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(class_path, img_name)
                try:
                    emb_new = get_embedding(img_path)   # koristi novi utils
                    if emb_new is None:
                        continue
                    label, dist = recognize(emb_new, thr=0.7)
                    print(f"{img_name};{class_name};{label};{dist:.3f}")
                except Exception as e:
                    print(f"Osoba: {class_name} {img_name}, Error: {e}")

def intra_class_test(person: str = "Mathias", thr: float = 0.75, max_images=30):
    mean_embs = np.load("centroid_znacajke.npy")
    mean_labels = np.load("oznake.npy")
    
    person_idx = np.where(mean_labels == person)[0]
    if len(person_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return
    
    centroid = mean_embs[person_idx[0]]
    
    person_path = os.path.join("dataset/train", person)
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    similarities = []
    correct = 0
    tested = 0

    print(f"Testiram {person} | Threshold: {thr} | Centroid norm: {np.linalg.norm(centroid):.4f}\n")

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb = get_embedding(img_path)   # koristi utils verziju
        
        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue
            
        sim = float(np.dot(centroid, emb))
        similarities.append(sim)
        
        label = person if sim > thr else "Unknown"
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
# show_face i ostali testovi mogu ostati zakomentirani ili ažurirani kasnije


if __name__ == "__main__":
    #similarities = intra_class_test("Mathias", thr=0.68, max_images=100)
    facial_recognition(thr=0.65)