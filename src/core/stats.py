# recognition/stats.py
from collections import Counter, defaultdict
from typing import Optional, Dict
import time
import numpy as np

class RecognitionStats:
    def __init__(self, temporal_window):
        self.temporal_window = temporal_window
        
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
        self.system_genuine_attempts = 0
        self.system_impostor_attempts = 0
        self.system_embedding = []
        self.system_false_acceptances = 0
        self.system_false_rejections = 0
        
        # Vremena
        self.detection_times = []
        self.embedding_times = []
        
        # Prozor za trenutno prepoznavanje
        self.current_embeddings = []
        self.current_window = Counter()
        self.system_recognized_person = "Unknown"
        
        self.start_time = time.perf_counter()

    def update(self, embedding, label: str, ground_truth: Optional[str] = None):
        """Glavna metoda za ažuriranje"""
        self.frame_count += 1
        self.total_labels[label] += 1
        
        if label == "Unknown":
            self.unknown_count += 1
            
        # Dodaj u trenutni prozor
        self.current_window[label] += 1
        self.current_embeddings.append(embedding)
        
        # Frame-level FAR/FRR
        if ground_truth is not None:
            self._update_frame_metrics(label, ground_truth)
        
        # System-level odluka svakih N frameova
        if self.frame_count % self.temporal_window == 0:
            self._make_system_embedding()
            self.current_window.clear()
            self.current_embeddings = []

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

    def _make_system_embedding(self):
        """System-level konsenzus"""
        if not self.current_window:
            return    
        if not self.current_embeddings:
            return
        self.system_embedding = np.median(self.current_embeddings, axis=0)

    def _make_system_decision(self, person: str, ground_truth: Optional[str] = None):
        """System-level odluka"""
        self.system_decisions += 1
        self.system_counter[person] += 1
        self.system_recognized_person = person
        self.system_embedding = []

        # Ovdje možeš dodati ground_truth za system-level FAR/FRR ako ga proslijediš
        if ground_truth is not None:
            if ground_truth != "Unknown":  # genuine pokušaj
                self.system_genuine_attempts += 1
                if person != ground_truth:
                    self.system_false_rejections += 1
            else:  # impostor
                self.system_impostor_attempts += 1
                if person != "Unknown":
                    self.system_false_acceptances += 1

    def add_detection_time(self, time_taken: float):
        self.detection_times.append(time_taken)

    def add_embedding_time(self, time_taken: float):
        self.embedding_times.append(time_taken)

    def get_system_recognized_person(self) -> str:
        return self.system_recognized_person
    
    def get_system_embedding(self):
        return self.system_embedding

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

    def get_system_far_frr(self) -> Dict:   # kasnije implementiraj s ground truth-om
        far = (self.system_false_acceptances / self.system_impostor_attempts * 100) if self.system_impostor_attempts > 0 else 0
        frr = (self.system_false_rejections / self.system_genuine_attempts * 100) if self.system_genuine_attempts > 0 else 0
        return {
            "FAR_system_%": round(far, 4),
            "FRR_system_%": round(frr, 4),
            "genuine_attempts": self.system_genuine_attempts,
            "impostor_attempts": self.system_impostor_attempts
        }

    def print_report(self):
        total = self.frame_count
        unknown_pct = (self.unknown_count / total * 100) if total > 0 else 0
        
        if len(self.system_counter) == 1:
            recognized = self.system_counter.most_common(1)[0][0]
        elif len(self.system_counter) > 1:
            recognized = self.system_counter.most_common(1)[0][0] if self.system_counter.most_common(1)[0][0] != "Unknown" else self.system_counter.most_common()[1][0]
        else:
            recognized = "Unknown"
        print("="*40)
        print("SUSTAV PREPOZNAO OSOBU:", recognized)
        print("="*40)

        print("\n=== RECOGNITION STATISTICS ===")
        print(f"Ukupno frameova: {total}")
        
        print(f"Unknown rate: {unknown_pct:.2f}%")
        print(f"System decisions: {self.system_decisions}")
        print("\nSustav prepoznao osobe:")
        for person, count in self.system_counter.most_common():
            print(f"  {person}: {count} puta")
        print("\nNajčešće prepoznate osobe:")
        for person, count in self.total_labels.most_common():
            print(f"  {person}: {count} frameova")
        
        print("\nFrame-level metrike:")
        frame_metrics = self.get_frame_far_frr()
        print(f"  FAR (frame): {frame_metrics['FAR_frame_%']}%")
        print(f"  FRR (frame): {frame_metrics['FRR_frame_%']}%")
        
        print("\nSystem-level metrike:")
        system_metrics = self.get_system_far_frr()
        print(f"  FAR (system): {system_metrics['FAR_system_%']}%")
        print(f"  FRR (system): {system_metrics['FRR_system_%']}%")
        
        print(f"\nProsječno vrijeme detekcije : {np.mean(self.detection_times)*1000:.2f} ms")
        print(f"Prosječno vrijeme embeddinga: {np.mean(self.embedding_times)*1000:.2f} ms")
