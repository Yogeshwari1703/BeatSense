import numpy as np
from scipy import signal

class ECGWearableProcessor:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
        print("ECG Wearable Processor initialized")
    
    def remove_baseline_wander(self, ecg_signal):
        from scipy import signal
        nyquist = self.fs / 2
        b, a = signal.butter(4, 0.5/nyquist, btype='high')
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered
    
    def suppress_motion_artifacts(self, ecg_signal):
        from scipy import signal
        low = 0.5 / (self.fs/2)
        high = 40 / (self.fs/2)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered
    
    def compute_signal_quality_index(self, ecg_signal):
        signal_std = np.std(ecg_signal)
        if signal_std < 0.01:
            return 0.1
        if signal_std > 2.0:
            return 0.3
        quality = min(1.0, signal_std / 0.5)
        return quality
    
    def process_signal(self, raw_signal):
        signal_bw_removed = self.remove_baseline_wander(raw_signal)
        cleaned_signal = self.suppress_motion_artifacts(signal_bw_removed)
        quality = self.compute_signal_quality_index(cleaned_signal)
        return cleaned_signal, quality