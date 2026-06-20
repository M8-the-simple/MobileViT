# detectors.py
import cv2
from facenet_pytorch import MTCNN  # zadrži za desktop testiranje

class FaceDetector:
    def __init__(self, method="haar", device=None):
        self.method = method
        self.mtcnn = None
        self.haar_cascade = None
        
        if method == "haar":
            self.haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if self.haar_cascade.empty():
                raise IOError("Nije učitan Haar cascade!")
        elif method == "mtcnn":
            self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False, 
                             select_largest=False, min_face_size=40)
    
    def detect(self, frame_bgr):
        """Vraća listu bounding boxova [(x1,y1,x2,y2), ...]"""
        if self.method == "haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) ##Boja nam nije bitna
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            # pretvori u (x1,y1,x2,y2)
            return [(x, y, x+w, y+h) for (x,y,w,h) in faces]
        else:  # mtcnn
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(frame_rgb)
            if boxes is None:
                return [], []
            return [box.astype(int).tolist() for box in boxes], probs
    
    def detect_and_crop(self, frame_bgr, min_confidence=0.5):
        if self.method == "mtcnn":
            boxes, probs = self.detect(frame_bgr)
            
            if boxes is None or probs is None or len(boxes) == 0:
                return None
            
            # Filtriranje po confidence
            valid = [(b, p) for b, p in zip(boxes, probs) 
                    if p is not None and p >= min_confidence]
            
            if not valid:
                return None
            
            # Uzmi najbolje lice
            best_box, best_prob = max(valid, key=lambda x: x[1])
            print(f"[DEBUG] Najbolji confidence: {best_prob:.4f}")  # ← korisno za debug
        elif self.method == "haar":
            boxes = self.detect(frame_bgr)
        if not boxes:
            print("Nisam uspio očitati bounding boxeve, model: ", self.method)
            return None
        
        # za Haar nemamo confidence → uzimamo najveće lice
        best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = best_box
        
        # margin
        h, w = frame_bgr.shape[:2]
        margin_h = int(0.2 * (y2 - y1))
        margin_w = int(0.2 * (x2 - x1))
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        return frame_bgr[y1:y2, x1:x2]