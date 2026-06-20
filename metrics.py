# metrics.py ili unutar utils.py
from collections import Counter
import time

class RecognitionStats:
    def __init__(self):
        self.label_count = Counter()
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.unknown_count = 0
    
    def update(self, label: str):
        self.label_count[label] += 1
        self.frame_count += 1
        if label == "Unknown":
            self.unknown_count += 1
    
    def print_report(self):
        print("\n" + "="*40)
        print("STATISTIKA PREPOZNAVANJA")
        print("="*40)
        
        total = self.frame_count
        for label, count in self.label_count.most_common():
            perc = (count / total * 100) if total > 0 else 0
            print(f"{label:12} → {count:4d}  ({perc:6.1f}%)")
        
        unknown_rate = (self.unknown_count / total * 100) if total > 0 else 0
        print(f"Unknown rate: {unknown_rate:.1f}%")
        print(f"Ukupno frame-ova: {total}")
        print(f"Vrijeme obrade: {time.perf_counter() - self.start_time:.1f} sekundi")