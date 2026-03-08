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


model, device = get_embedding_model()
mtcnn = MTCNN(keep_all=True, device=device)
transform = get_transform()


mean_embs = np.load("centroid_znacajke.npy")
mean_labels = np.load("oznake.npy")


# NOVA funkcija - transform tensor direktno



def recognize(emb_new, thr=0.85):
    sim = np.dot(mean_embs, emb_new)
    idx = np.argmax(sim)
    if sim[idx] > thr:
        return mean_labels[idx], sim[idx]
    else:
        return "Unknown", sim[idx]
def validation_test():
    for class_name in os.listdir("dataset/val"):
        class_path = os.path.join("dataset/val", class_name)
        if not os.path.isdir(class_path):
            continue
       
        for i, img_name in enumerate(os.listdir(class_path)):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, img_name)
            try:
                emb_new = get_embedding(img_path)
                label, dist = recognize(emb_new, thr=0.7)
                print(f"Osoba: {class_name} {img_name} Rezultat: {label} ({dist:.3f})")
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


# def test_embeddings():
#     face_test_path = "dataset/train/Mathias"
#     obj_test_path = "imagenet-val/n01440764"
#     face_paths = os.listdir(face_test_path)
#     obj_paths = os.listdir(obj_test_path)
#     face_embs = [(mtcnn(Image.open(os.path.join(face_test_path, face_path)))) for face_path in face_paths if face_path.lower().endswith((".png", ".jpg", ".jpeg"))]
#     obj_embs = [get_embedding(os.path.join(obj_test_path, obj_path)) for obj_path in obj_paths]
#     print(f"Središnja vrijednost lica: {np.mean(face_embs, axis=0)[:5]}...")
#     print(f"Središnja vrijednost objekata: {np.mean(obj_embs, axis=0)[:5]}...")


def test_intra_class():
    mean_emb_path = "dataset_faces/train/Mathias"
    test_embs = []
    for face_paths in os.listdir(mean_emb_path):
        emb = get_embedding(os.path.join(mean_emb_path, face_paths))
        test_embs.append(emb)
    if len(test_embs) < 2:
        print("Nedovoljno testnih slika za izračun sličnosti.")
        return
    mean_emb = np.mean(test_embs, axis=0)
    face_test_path = "dataset/val/Mathias"
    for face_path in os.listdir(face_test_path):
        #crop_emb = get_embedding_from_crop(mtcnn(Image.open(os.path.join(face_test_path, face_path))))
        print(f"Testna slika: {face_path}, Sličnost sa srednjom vrijednošću: {np.dot(mean_emb):.3f}")


def get_embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)        
    emb_np = emb.cpu().numpy()[0]  # (1, D)
    return emb_np / np.linalg.norm(emb_np)  # Normalizacija vektora


# def same_picture_two_methods():
#     img_path = "dataset_faces/val/Mathias/00044.png"


#     faces = mtcnn(Image.open(img_path))
   
#     plt.figure(figsize=(12, 8))
#     plt.imshow(faces[0].permute(1, 2, 0).int().numpy())
#     plt.axis("off")
#     plt.show()


#     #emb_from_crop = get_embedding_from_crop(faces)
#     emb_from_full = get_embedding(img_path)
#     print(f"Sličnost između metoda: {np.dot(emb_from_full):.3f}")
   
def facial_recognition():
    
    cap = FileVideoStream("output.avi").start()
    v_len = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    fps_text = ''
    text = ''
    fps = 0
    start_time = time.perf_counter()


    for _ in range(v_len):
        frame = cap.read()


        frame_count +=1


        #if not ret:
        #   break

        before_detection = time.perf_counter()
        faces, probs = mtcnn.detect(frame)
        after_detection = time.perf_counter()

        print(f"Faces: {faces.shape}")

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


                face_crop = frame[y1:y2, x1:x2]  # i dalje BGR, što je ok za get_embedding_from_crop
               
           
                try:
                    before_embedding = time.perf_counter()
                    emb_new = get_embedding_from_crop(face_crop)
                    after_embedding = time.perf_counter()
                    #print(f"Vrijeme za embedding: {(time.perf_counter() - before_embedding)*1000:.1f} ms")
                    before_recognition = time.perf_counter()
                    label, dist = recognize(emb_new, thr=0.7)
                    after_recognition = time.perf_counter()
                    text = f"{label} ({dist:.2f})"
                except Exception as e:
                    text = f"error: {str(e)}"
                    #print(text)


                    # iscrtaj bounding box i label
                before_draw = time.perf_counter()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                after_draw = time.perf_counter()
                detection = after_detection - before_detection
                embedding = after_embedding - before_embedding
                recognition = after_recognition - before_recognition
                draw = after_draw - before_draw

                


        fps_text = f"FPS: {fps:.1f}"
        if frame_count % 30 == 0:
            now = time.perf_counter()
            elapsed = now - start_time
            print(f"Vrijeme detekcije: {detection*1000:.1f} ms, Vrijeme embeddinga: {embedding*1000:.1f} ms, Vrijeme prepoznavanja: {recognition*1000:.1f} ms, Vrijeme crtanja: {draw*1000:.1f} ms, Sveukupno: {elapsed*1000/30:.1f} ms")
            fps = 30.0 / elapsed
            start_time = now


        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


         # Detekcija lica


        cv2.imshow("Facial recognition", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #test_intra_class()
    #same_picture_two_methods()
    #show_face()
    #main()
    facial_recognition()
    #validation_test()
