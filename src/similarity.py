import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from src.common.config import get_config
from src.common.factory import ComponentFactory

cfg = get_config()
model = ComponentFactory.create_model()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()

img = cv2.imread(r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\dataset\test\Antonio\frame_000032_conf0.9967.jpg")

embs = []

for i in range(100):

    boxes, probs, landmarks = detector.detect(img)

    preprocessed_img = preprocessor(img, landmarks=landmarks[0] if len(landmarks) > 0 else None)

    emb = model.embed(preprocessed_img)
    embs.append(emb)
    if len(embs) > 1:
        sim = np.dot(embs[-1], embs[-2])
        print(f"Iteracija {i+1}: Sličnost između ova dva embeddinga je: {sim:.6f}")

# for person_img in os.listdir("dataset/val/Matej"):
    
#     if not person_img.lower().endswith(('.jpg', '.jpeg', '.png')):
#         continue
#     img_path = os.path.join("dataset/val/Matej", person_img)

#     img = cv2.imread(img_path)

#     boxes, probs, landmarks = detector.detect(img)
#     faces = [(b, p, l) for b, p, l in zip(boxes, probs, landmarks)]
#     if len(faces) == 0:
#         print(f"Nije detektirano lice na slici: {img_path}")
#         continue
#     highest_conf_face = max(faces, key=lambda x: x[1] if x[1] is not None else 0)
#     box, prob, landmarks = highest_conf_face

#     preprocessed_img = preprocessor(img, landmarks=landmarks)

#     plt.imshow(preprocessed_img)
#     plt.show()

# emb1 = get_embedding(r"dataset\val\Antonio\frame_000020_conf0.9723.jpg")
# emb2 = get_embedding(r"dataset\val\David\frame_000048_conf0.9821.jpg")

# print(f"Sličnost između ova dva embeddinga je: {np.dot(emb1, emb2)}")

# cap = cv2.VideoCapture(r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\val_videos\Antonio\video_Antonio_20260518_184526.mp4")

# if not cap.isOpened():
#     print("Nemogu otvoriti video")
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print("Nema više frame-ova ili greška u čitanju")
#         break
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()