import cv2
from facenet_pytorch import MTCNN
from Embedding_model import get_embedding_model, get_transform
import numpy as np
from PIL import Image
import torch
import os


model, device = get_embedding_model()
transform = get_transform()
mtcnn = MTCNN(keep_all=True, device=device)

def detect_and_crop(img):
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
            for box, p in zip(boxes, probs):
                if p is None:
                    continue
                x1, y1, x2, y2 = box.astype(int)

                # sigurnosna rezanja da ne izađemo iz slike
                h_frame, w_frame, _ = img.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_frame, x2)
                y2 = min(h_frame, y2)

                face_crop = img[y1:y2, x1:x2]  # i dalje BGR, što je ok za get_embedding_from_crop
                return face_crop
    return img  # Ako nema detekcije, vrati original


def get_embedding(input):
    if os.path.exists(input):
        img = cv2.imread(input)
    else:
        img = input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb is None:
        raise ValueError(f"Ne mogu otvoriti sliku: {input}")
    img_cropped = detect_and_crop(img_rgb)  # Detekcija i isjecanje lica (ako je potrebno)
    pil_img = Image.fromarray(img_cropped)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    emb_np = emb.cpu().numpy().flatten()
    return emb_np / np.linalg.norm(emb_np)  # Normalizacija vektora