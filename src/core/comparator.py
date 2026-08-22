import numpy as np
from typing import Dict, Tuple, Optional

class CentroidComparator:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, thresholds: Dict[str, float]):
        self.embeddings = embeddings  # shape: (n_people, embedding_dim)
        self.labels = labels         # shape: (n_people,)
        self.thresholds = thresholds

    def compare(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Compare embedding against all centroids."""
        similarities = np.dot(self.embeddings, embedding)
        idx = np.argmax(similarities)

        label = self.labels[idx]
        threshold = 0.5

        if similarities[idx] >= threshold:
            return label, float(similarities[idx])
        return "Unknown", float(similarities[idx])