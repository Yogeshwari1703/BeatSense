import numpy as np

class ArrhythmiaDetector:
    def __init__(self):
        self.arrhythmia_types = {
            0: "Normal Sinus Rhythm",
            1: "Atrial Fibrillation",
            2: "Ventricular Tachycardia",
            3: "Bradycardia"
        }
        print("Arrhythmia Detector initialized")
    
    def extract_features(self, signal):
        features = {}
        if len(signal) < 10:
            return None
        
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        
        threshold = np.std(signal) * 0.5
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks)
            features['heart_rate'] = 60 / (np.mean(rr_intervals) / 360)
            features['hrv'] = np.std(rr_intervals)
        else:
            features['heart_rate'] = 72
            features['hrv'] = 0.1
        
        features['signal_quality'] = min(1.0, np.std(signal) / 0.5)
        
        return features
    
    def predict(self, signal, quality_score=1.0):
        features = self.extract_features(signal)
        
        if features is None:
            return {
                'arrhythmia_type': 'Unknown',
                'confidence': 0.3,
                'signal_quality': 0,
                'reliable': False,
                'heart_rate': 0
            }
        
        hr = features.get('heart_rate', 72)
        quality = features.get('signal_quality', 0.5)
        
        if hr < 60:
            diagnosis = "Bradycardia"
            confidence = 0.7
        elif hr > 100:
            diagnosis = "Tachycardia"
            confidence = 0.7
        else:
            diagnosis = "Normal Sinus Rhythm"
            confidence = 0.8
        
        return {
            'arrhythmia_type': diagnosis,
            'confidence': confidence * quality,
            'signal_quality': quality,
            'heart_rate': hr,
            'reliable': quality > 0.6
        }