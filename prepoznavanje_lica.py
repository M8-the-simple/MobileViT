import time

from models import get_embedding_model, get_transform, get_detector
from PIL import Image
import os
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_embedding, load_centroids
from imutils.video import FPS, FileVideoStream
from tqdm import tqdm
import random
from metrics import RecognitionStats
from tests import intra_class_test, inter_class_test


model, device = get_embedding_model()
transform = get_transform()

detector = get_detector()   # Unutar manager.py se odlučuje koji detektor se koristi

mean_embs, mean_labels = load_centroids(model)


def recognize(emb_new):
    from config import THRESHOLDS

    sim = np.dot(mean_embs, emb_new)
    idx = np.argmax(sim)
    if sim[idx] > THRESHOLDS[model.name][mean_labels[idx]]:
        print("Unutar if-a")
        return mean_labels[idx], sim[idx]
    else:
        return "Unknown", sim[idx]

def facial_recognition():
    #cap = cv2.VideoCapture(0)  # Koristi kameru
    cap = cv2.VideoCapture("video_Matej_20260518_165925.mp4")  # za testiranje videa
    #cap = cv2.VideoCapture("video_Mathias_20260518_163440.mp4")  # za testiranje videa
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
                emb_new = get_embedding(face_crop)   # ili get_embedding(face_crop)
                after_embedding = time.perf_counter()
                before_recognition = time.perf_counter()
                print("[DEBUG] Došli smo do try blocka")
                label, dist = recognize(emb_new)
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



def facial_recognition_insightface():
    #cap = cv2.VideoCapture(0)  # Koristi kameru
    cap = cv2.VideoCapture("video_Matej_20260518_165925.mp4")  # za testiranje videa
    #cap = cv2.VideoCapture(r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\Osobe\Antonio\video_Antonio_20260518_184548.mp4")  # za testiranje videa
    
    frame_count = 0
    fps_text = ''
    text = ''
    embedding = 0
    recognition = 0
    avg_embedding = []
    stats = RecognitionStats()
    start_time = time.perf_counter()
    fps_imutils = FPS().start()
    fps = 0

    if not cap.isOpened():
        print("Nemogu otvoriti kameru")
        return

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if frame is None:
            print("Nisam u mogućnosti pročitati frame")
            break

        try:
            # --- Generisanje embeddinga za ceo frame ---
            before_embedding = time.perf_counter()
            emb_new = get_embedding(frame)   
            after_embedding = time.perf_counter()

            # --- Prepoznavanje ---
            before_recognition = time.perf_counter()
            label, dist = recognize(emb_new) 
            stats.update(label)
            after_recognition = time.perf_counter()

            text = f"{label} ({dist:.2f})"
            embedding = after_embedding - before_embedding
            recognition = after_recognition - before_recognition

        except Exception as e:
            text = "error"
            embedding = 0
            recognition = 0

        # --- Crtanje rezultata (sada ispisujemo tekst u gornjem levom uglu) ---
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        avg_embedding.append(embedding)

        # --- FPS i logiranje ---
        fps_imutils.update()
        fps_text = f"FPS: {fps:.1f}"

        if frame_count % 30 == 0 and frame_count > 0:
            now = time.perf_counter()
            elapsed = now - start_time
            fps = 30 / elapsed
            print(f"Embedding: {embedding*1000:.1f} ms | "
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
    if avg_embedding:
        print(f"Prosječno vrijeme embeddinga: {np.mean(avg_embedding)*1000:.1f} ms")
    stats.print_report()

def get_embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Prilagođeno za InsightFace"""
    if hasattr(model, 'app'):   # InsightFace
        return get_embedding(crop_bgr)   # koristi glavnu funkciju
    
    # stari put...
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)        
    emb_np = emb.cpu().numpy()[0]
    return emb_np / np.linalg.norm(emb_np)


# === Testiranje s nasumičnim osobama iz skupa podataka ===
# def validation_test():
#     for dataset_path in ["dataset/random"]:
#         for class_name in os.listdir(dataset_path):
#             class_path = os.path.join(dataset_path, class_name)
#             if not os.path.isdir(class_path):
#                 continue
#             for i, img_name in enumerate(os.listdir(class_path)):
#                 if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
#                     continue
#                 img_path = os.path.join(class_path, img_name)
#                 try:
#                     emb_new = get_embedding(img_path)   # koristi novi utils
#                     if emb_new is None:
#                         continue
#                     label, dist = recognize(emb_new, thr=0.7)
#                     print(f"{img_name};{class_name};{label};{dist:.3f}")
#                 except Exception as e:
#                     print(f"Osoba: {class_name} {img_name}, Error: {e}")


if __name__ == "__main__":
    #for person in os.listdir(r"dataset\train"):
    #   similarities = intra_class_test(person, thr=0.5, max_images=40, mean_embeddings=mean_embs, mean_labels=mean_labels)
    facial_recognition()