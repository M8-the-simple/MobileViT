import numpy as np
from typing import Dict, Tuple, Optional

class CentroidComparator:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, thresholds: Dict[str, float]):
        self.embeddings = embeddings  # shape: (n_people, embedding_dim)
        self.labels = labels         # shape: (n_people,)
        self.thresholds = thresholds

    def compare(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Compare embedding against all centroids."""
        if embedding is None:
            return "Unknown", 0.0
        similarities = np.dot(self.embeddings, embedding)
        
        idx = int(np.argmax(similarities))
        label = self.labels[idx]
        threshold = float(self.thresholds.get(label, 0.5)) if self.thresholds else 0.5
        score = float(similarities[idx])
        if score >= threshold:
            return label, score
        else:
            return "Unknown", score