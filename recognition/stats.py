# recognition/stats.py
from collections import Counter, defaultdict
from typing import Optional, Dict
import time
import numpy as np

class RecognitionStats:
    def __init__(self, config=None):
        self.config = config or {}
        self.temporal_window = getattr(self.config, 'temporal_window', 5)
        
        # --- Frame-level statistike ---
        self.frame_count = 0
        self.total_labels = Counter()           # Sve label-e preko svih frameova
        self.unknown_count = 0
        
        # Za FAR/FRR (potreban ground truth!)
        self.genuine_attempts = 0      # koliko frameova gdje je ground_truth = true_person
        self.false_rejections = 0      # true_person → Unknown ili kriva osoba
        self.false_acceptances = 0     # impostor → prihvaćen kao netko
        self.impostor_attempts = 0
        
        # --- System-level (nakon konsenzusa) ---
        self.system_decisions = 0
        self.system_counter = Counter()
        self.system_false_accept = 0
        self.system_false_reject = 0
        
        # Vremena
        self.detection_times = []
        self.embedding_times = []
        
        # Prozor za trenutno prepoznavanje
        self.current_window = Counter()
        self.system_recognized_person = "Unknown"
        
        self.start_time = time.perf_counter()

    def update(self, label: str, ground_truth: Optional[str] = None):
        """Glavna metoda za ažuriranje"""
        self.frame_count += 1
        self.total_labels[label] += 1
        
        if label == "Unknown":
            self.unknown_count += 1
            
        # Dodaj u trenutni prozor
        self.current_window[label] += 1
        
        # Frame-level FAR/FRR
        if ground_truth is not None:
            self._update_frame_metrics(label, ground_truth)
        
        # System-level odluka svakih N frameova
        if self.frame_count % self.temporal_window == 0:
            self._make_system_decision()
            self.current_window.clear()

    def _update_frame_metrics(self, predicted: str, ground_truth: str):
        """Frame-level metrike"""
        if ground_truth != "Unknown":  # genuine pokušaj
            self.genuine_attempts += 1
            if predicted != ground_truth:
                self.false_rejections += 1
        else:  # impostor
            self.impostor_attempts += 1
            if predicted != "Unknown":
                self.false_acceptances += 1

    def _make_system_decision(self):
        """System-level konsenzus"""
        if not self.current_window:
            return
            
        most_common = self.current_window.most_common(1)
        if not most_common:
            return
            
        system_label = most_common[0][0]
        self.system_counter[system_label] += 1
        self.system_decisions += 1
        self.system_recognized_person = system_label

        # Ovdje možeš dodati ground_truth za system-level FAR/FRR ako ga proslijediš

    def add_detection_time(self, time_taken: float):
        self.detection_times.append(time_taken)

    def add_embedding_time(self, time_taken: float):
        self.embedding_times.append(time_taken)

    def get_system_recognized_person(self) -> str:
        return self.system_recognized_person

    # === METRIKE ===
    def get_frame_far_frr(self) -> Dict:
        far = (self.false_acceptances / self.impostor_attempts * 100) if self.impostor_attempts > 0 else 0
        frr = (self.false_rejections / self.genuine_attempts * 100) if self.genuine_attempts > 0 else 0
        return {
            "FAR_frame_%": round(far, 4),
            "FRR_frame_%": round(frr, 4),
            "genuine_attempts": self.genuine_attempts,
            "impostor_attempts": self.impostor_attempts
        }

    # def get_system_far_frr(self) -> Dict:   # kasnije implementiraj s ground truth-om
    #     return {"note": "System-level FAR/FRR još nije implementiran (potreban GT po videu)"}

    def print_report(self):
        total = self.frame_count
        unknown_pct = (self.unknown_count / total * 100) if total > 0 else 0
        
        print("="*40)
        print("SUSTAV PREPOZNAO OSOBU:", self.system_counter.most_common(1)[0][0] if self.system_counter.most_common(1)[0][0] != "Unknown" else self.system_counter.most_common()[1][0])
        print("="*40)

        print("\n=== RECOGNITION STATISTICS ===")
        print(f"Ukupno frameova: {total}")
        print(f"Unknown rate: {unknown_pct:.2f}%")
        print(f"System decisions: {self.system_decisions}")
        print("\nNajčešće prepoznate osobe:")
        for person, count in self.total_labels.most_common():
            print(f"  {person}: {count} frameova")
        
        print("\nFrame-level metrike:")
        metrics = self.get_frame_far_frr()
        print(f"  FAR (frame): {metrics['FAR_frame_%']}%")
        print(f"  FRR (frame): {metrics['FRR_frame_%']}%")
        
        print(f"\nProsječno vrijeme detekcije : {np.mean(self.detection_times)*1000:.2f} ms")
        print(f"Prosječno vrijeme embeddinga: {np.mean(self.embedding_times)*1000:.2f} ms")