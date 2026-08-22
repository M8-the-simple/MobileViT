# tests package
import cv2
import numpy as np
import time
from src import ComponentFactory

detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()
model = ComponentFactory.create_model()

def get_embedding_from_image(image_path):
    """Load image, detect face, preprocess, and extract embedding.
    Returns (embedding, elapsed_time) or (None, None) if no face detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️  Cannot read image: {image_path}")
        return None, None


    if model.name.startswith("buffalo_sc"):
        before_emb = time.perf_counter()
        embedding = model.embed(img)
        elapsed_emb = time.perf_counter() - before_emb
        if embedding is not None:
            return embedding, elapsed_emb
        else:
            print(f"  ⚠️  No face detected in image: {image_path}")
            return None, None

    boxes, probs, landmarks = detector.detect(img)

    if len(boxes) == 0:
        return None, None

    # Take highest confidence face
    best_idx = np.argmax(probs) if len(probs) > 0 else 0
    box = boxes[best_idx]
    land = landmarks[best_idx] if landmarks is not None and len(landmarks) > best_idx else None

    # Preprocess
    processed = preprocessor(img, land)
    if processed is None:
        return None, None

    # Extract embedding
    before_emb = time.perf_counter()
    emb = model.embed(processed)
    elapsed_emb = time.perf_counter() - before_emb
    return emb, elapsed_emb
