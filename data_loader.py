"""
Load and preprocess REAL MIT-BIH Arrhythmia Database
This uses the ACTUAL dataset, not simulations
"""

import wfdb
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
import os


class MITBIHLoader:
    """
    Loader for REAL MIT-BIH Arrhythmia Database
    This is the EXACT dataset used in your paper's references
    """
    
    def __init__(self, data_path='data/mitbih'):
        self.data_path = data_path
        self.sampling_rate = 360  # MIT-BIH uses 360 Hz
        self.class_mapping = {
            # Normal beats
            'N': 0, 'L': 0, 'R': 0,  # Normal, Left bundle, Right bundle
            # Supraventricular ectopic beats
            'A': 1, 'a': 1, 'J': 1, 'S': 1,  # Atrial premature, Aberrated, Nodal, Supraventricular
            # Ventricular ectopic beats
            'V': 2, 'E': 2,  # Ventricular, Ventricular escape
            # Fusion beats
            'F': 3,  # Fusion of ventricular and normal
            # Unknown/Other
            '/': 4, 'Q': 4  # Paced, Unknown
        }
        
        # Complete list of records
        self.records = [
            '100', '101', '102', '103', '104', '105', '106', '107', 
            '108', '109', '111', '112', '113', '114', '115', '116',
            '117', '118', '119', '121', '122', '123', '124', '200',
            '201', '202', '203', '205', '207', '208', '209', '210',
            '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        
    def load_record(self, record_name):
        """
        Load a single MIT-BIH record
        Returns: signal, annotations
        """
        try:
            # Load signal
            record = wfdb.rdrecord(f'{self.data_path}/{record_name}')
            # Load annotations (beat labels)
            annotation = wfdb.rdann(f'{self.data_path}/{record_name}', 'atr')
            
            print(f"  Loaded {record_name}: {record.p_signal.shape[0]} samples, {len(annotation.symbol)} beats")
            
            return record, annotation
            
        except Exception as e:
            print(f"  Error loading {record_name}: {e}")
            return None, None
    
    def segment_ecg(self, signal_data, annotation, window_size=3600):
        """
        Segment ECG into windows with labels
        window_size = 3600 samples = 10 seconds at 360 Hz
        """
        windows = []
        labels = []
        
        # Get beat positions and types
        beat_positions = annotation.sample
        beat_types = annotation.symbol
        
        # Create segments
        for i in range(0, len(signal_data) - window_size, window_size//2):  # 50% overlap
            window = signal_data[i:i+window_size]
            
            # Find beats in this window
            beats_in_window = [j for j, pos in enumerate(beat_positions) 
                              if i <= pos < i+window_size]
            
            if len(beats_in_window) > 0:
                # Get most common beat type in this window
                beat_type_counts = {}
                for beat_idx in beats_in_window:
                    beat_type = beat_types[beat_idx]
                    if beat_type in self.class_mapping:
                        label = self.class_mapping[beat_type]
                        beat_type_counts[label] = beat_type_counts.get(label, 0) + 1
                
                if beat_type_counts:
                    # Majority vote for window label
                    window_label = max(beat_type_counts, key=beat_type_counts.get)
                    
                    # Normalize window
                    window = (window - np.mean(window)) / (np.std(window) + 1e-6)
                    
                    windows.append(window)
                    labels.append(window_label)
        
        return np.array(windows), np.array(labels)
    
    def load_all_data(self, max_records=None):
        """
        Load and process ALL MIT-BIH records
        This is the COMPLETE dataset as used in your paper
        """
        all_windows = []
        all_labels = []
        
        records_to_load = self.records[:max_records] if max_records else self.records
        
        print(f"\n📚 Loading {len(records_to_load)} MIT-BIH records...")
        print("="*60)
        
        for i, record_name in enumerate(records_to_load):
            print(f"[{i+1}/{len(records_to_load)}] ", end="")
            
            # Load record
            record, annotation = self.load_record(record_name)
            if record is None:
                continue
            
            # Get signal (use MLII lead)
            ecg_signal = record.p_signal[:, 0]
            
            # Apply bandpass filter (0.5-40 Hz) to remove noise
            nyquist = self.sampling_rate / 2
            b, a = scipy_signal.butter(4, [0.5/nyquist, 40/nyquist], btype='band')
            filtered_signal = scipy_signal.filtfilt(b, a, ecg_signal)
            
            # Segment
            windows, labels = self.segment_ecg(filtered_signal, annotation)
            
            all_windows.extend(windows)
            all_labels.extend(labels)
            
            print(f"  → Added {len(windows)} segments")
        
        X = np.array(all_windows)
        y = np.array(all_labels)
        
        print("\n" + "="*60)
        print(f"✅ COMPLETE: Loaded {len(X)} ECG segments")
        print(f"   Input shape: {X.shape}")
        print(f"   Class distribution: {np.bincount(y)}")
        print("="*60)
        
        return X, y
    
    def get_train_test_split(self, X, y, test_size=0.2):
        """
        Create train/test split for model training
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    loader = MITBIHLoader()
    
    # Load first 10 records (takes ~2-3 minutes)
    X, y = loader.load_all_data(max_records=10)
    
    print(f"\n✅ REAL MIT-BIH data loaded!")
    print(f"   This is the actual dataset used in research")