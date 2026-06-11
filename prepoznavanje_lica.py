import time


from Embedding_model import get_embedding_model, get_transform
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import get_embedding
from imutils.video import FileVideoStream
from imutils.video import FPS


model, device = get_embedding_model()
mtcnn = MTCNN(keep_all=True, device=device, post_process=False, select_largest=False, min_face_size=40)
transform = get_transform()


mean_embs = np.load("centroid_znacajke.npy")
mean_labels = np.load("oznake.npy")

def recognize(emb_new, thr=0.85):
    sim = np.dot(mean_embs, emb_new)
    idx = np.argmax(sim)
    if sim[idx] > thr:
        return mean_labels[idx], sim[idx]
    else:
        return "Unknown", sim[idx]
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
                    emb_new = get_embedding(img_path)
                    if emb_new is None:
                        continue
                    label, dist = recognize(emb_new, thr=0.7)
                    print(f"{img_name};{class_name};{label};{dist:.3f}")
                except Exception as e:
                    print(f"Osoba: {class_name} {img_name} (face {i+1}), Error: {e}")
           
def show_face():
    import matplotlib.pyplot as plt
   
    img_path = "dataset_faces/val/Mathias/00044.png"
    img = cv2.imread(img_path)
   
    if img is None:
        print(f"Ne mogu procitati sliku: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)


    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    plt.axis('off')
    #plt.show()
   
    face = mtcnn(pil_img)


    face_tensor = face[0]


    plt.figure(figsize=(12, 8))
    plt.imshow(face_tensor.permute(1, 2, 0).int().numpy())
    plt.axis("off")
    plt.show()


def test_intra_class():
    mean_emb_path = "dataset/train/Fabris"
    for face_paths in os.listdir(mean_emb_path):
        emb = get_embedding(os.path.join(mean_emb_path, face_paths))
        if emb is None:
            continue
        label, probability = recognize(emb, thr=0.7)
        print(f"Testna slika: {face_paths}, Prepoznata osoba: {label}, Sličnost: {probability:.3f}")
        
def cross_similarity_test():
    persons = ['Alison', 'Mathias', 'Dominik', 'Fabris']
    for test_person in persons:
        print(f"\n=== {test_person} test ===")
        for img_name in os.listdir(f"dataset/train/{test_person}")[:5]:
            emb = get_embedding(f"dataset/train/{test_person}/{img_name}")
            sims = np.dot(mean_embs, emb)
            best_match = mean_labels[np.argmax(sims)]
            best_sim = np.max(sims)
            print(f"{img_name}: {best_match} ({best_sim:.3f})")

        #crop_emb = get_embedding_from_crop(mtcnn(Image.open(os.path.join(face_test_path, face_path))))


def get_embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)        
    emb_np = emb.cpu().numpy()[0]  # (1, D)
    return emb_np / np.linalg.norm(emb_np)  # Normalizacija vektora

   
def facial_recognition():
    
    #cap = FileVideoStream("output1.avi").start() ## Ako želimo koristiti video datoteku umjesto kamere
    #v_len = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    cap = cv2.VideoCapture(0)  # Koristi kameru
    frame_count = 0
    fps_text = ''
    text = ''
    avg_detection = []
    avg_embedding = []
    start_time = time.perf_counter()
    fps_imutils = FPS().start()
    fps = 0
    detection = 0
    embedding = 0
    recognition = 0
    draw = 0
    faces = None
    probs = None
    if not cap.isOpened():
        print("Nemogu otvoriti video")
        return

    #for _ in range(v_len): ## Ako želimo koristiti video datoteku umjesto kamere
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Nisam u mogućnosti pročitati frame")
            cap.release()
            return
        if frame is None:
            continue

        if frame_count % 5 == 0:
            before_detection = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, probs = mtcnn.detect(frame_rgb)
            after_detection = time.perf_counter()
            detection = after_detection - before_detection

        if faces is not None:
            for face, p in zip(faces, probs):
                if p is None:
                    continue
                x1, y1, x2, y2 = face.astype(int)


                # sigurnosna rezanja da ne izađemo iz slike
                h_frame, w_frame, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_frame, x2)
                y2 = min(h_frame, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
               
           
                try:
                    before_embedding = time.perf_counter()
                    emb_new = get_embedding_from_crop(face_crop)
                    after_embedding = time.perf_counter()
                    before_recognition = time.perf_counter()
                    label, dist = recognize(emb_new, thr=0.7)
                    after_recognition = time.perf_counter()
                    text = f"{label} ({dist:.2f})"
                    embedding = after_embedding - before_embedding
                    recognition = after_recognition - before_recognition
                except Exception as e:
                    text = f"error: {str(e)}"

                before_draw = time.perf_counter()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                after_draw = time.perf_counter()
                avg_detection.append(detection)
                avg_embedding.append(embedding)
                draw = after_draw - before_draw

                

        fps_imutils.update()
        fps_text = f"FPS: {fps:.1f}"
        if frame_count % 30 == 0 and frame_count > 0:
            now = time.perf_counter()
            elapsed = now - start_time
            fps = 30 / elapsed
            print(f"Vrijeme detekcije: {detection*1000:.1f} ms, Vrijeme embeddinga: {embedding*1000:.1f} ms, Vrijeme prepoznavanja: {recognition*1000:.1f} ms, Vrijeme crtanja: {draw*1000:.1f} ms, Sveukupno: {elapsed*1000/30:.1f} ms")
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
        print(f"Prosječno vrijeme detekcije lica : {np.mean(avg_detection)*1000:.1f} ms")
    if avg_embedding:
        print(f"Prosječno vrijeme embeddinga : {np.mean(avg_embedding)*1000:.1f} ms")


if __name__ == "__main__":
    facial_recognition()
    
