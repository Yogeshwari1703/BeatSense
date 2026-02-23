"""
Signal Quality Index (SQI) Implementation
Based on vital_sqi toolkit and your paper's methodology
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
import json

class SignalQualityIndex:
    """
    Implements multiple SQI metrics as described in your paper
    """
    
    def __init__(self, fs=360):
        self.fs = fs
        
    def compute_all_sqis(self, ecg_signal):
        """
        Compute ALL quality metrics
        Returns comprehensive quality assessment
        """
        sqis = {}
        
        # 1. Statistical SQIs
        sqis['skewness'] = self._skewness_sqi(ecg_signal)
        sqis['kurtosis'] = self._kurtosis_sqi(ecg_signal)
        sqis['entropy'] = self._entropy_sqi(ecg_signal)
        
        # 2. Signal-to-Noise Ratio
        sqis['snr'] = self._snr_sqi(ecg_signal)
        
        # 3. Baseline wander detection
        sqis['baseline'] = self._baseline_sqi(ecg_signal)
        
        # 4. Motion artifact detection
        sqis['motion'] = self._motion_sqi(ecg_signal)
        
        # 5. R-peak reliability
        sqis['rpeak'] = self._rpeak_sqi(ecg_signal)
        
        # 6. Powerline interference
        sqis['powerline'] = self._powerline_sqi(ecg_signal)
        
        # Calculate overall quality score
        weights = {
            'skewness': 0.1,
            'kurtosis': 0.1,
            'entropy': 0.15,
            'snr': 0.2,
            'baseline': 0.15,
            'motion': 0.15,
            'rpeak': 0.1,
            'powerline': 0.05
        }
        
        overall = sum(sqis[k] * weights[k] for k in weights if k in sqis)
        
        return {
            'overall_quality': overall,
            'detailed_sqis': sqis,
            'reliable': overall > 0.7,
            'acceptable': overall > 0.5
        }
    
    def _skewness_sqi(self, signal):
        """Skewness-based quality metric"""
        skew = stats.skew(signal)
        # Normalize: acceptable range [-0.5, 0.5]
        quality = 1 - min(1, abs(skew) / 2)
        return max(0, min(1, quality))
    
    def _kurtosis_sqi(self, signal):
        """Kurtosis-based quality metric"""
        kurt = stats.kurtosis(signal)
        # Normal kurtosis ~3, acceptable range [2, 5]
        if 2 <= kurt <= 5:
            return 1.0
        elif kurt < 2:
            return max(0, kurt / 2)
        else:  # kurt > 5
            return max(0, 1 - (kurt - 5) / 5)
    
    def _entropy_sqi(self, signal):
        """Sample entropy-based quality"""
        # Simplified entropy calculation
        r = 0.2 * np.std(signal)
        N = len(signal)
        
        # Count matches
        matches = 0
        for i in range(N-1):
            for j in range(i+1, N-1):
                if abs(signal[i] - signal[j]) < r and abs(signal[i+1] - signal[j+1]) < r:
                    matches += 1
        
        entropy = -np.log(matches / (N*(N-1)/2 + 1e-6) + 1e-6)
        # Normalize: lower entropy = better quality
        return max(0, min(1, 1 - entropy/5))
    
    def _snr_sqi(self, ecg_signal):
        """Signal-to-Noise Ratio quality"""
        # Estimate noise using high-frequency components
        b, a = scipy_signal.butter(4, 35/(self.fs/2), btype='high')
        noise = scipy_signal.filtfilt(b, a, ecg_signal)
        
        signal_power = np.var(ecg_signal)
        noise_power = np.var(noise) + 1e-6
        snr = 10 * np.log10(signal_power / noise_power)
        
        # Normalize: >20dB is excellent
        return min(1, max(0, snr / 20))
    
    def _baseline_sqi(self, ecg_signal):
        """Detect baseline wander"""
        # Get low-frequency component
        b, a = scipy_signal.butter(4, 0.5/(self.fs/2), btype='low')
        baseline = scipy_signal.filtfilt(b, a, ecg_signal)
        
        baseline_power = np.var(baseline)
        signal_power = np.var(ecg_signal) + 1e-6
        
        # Lower baseline power = better quality
        ratio = baseline_power / signal_power
        return max(0, min(1, 1 - ratio))
    
    def _motion_sqi(self, ecg_signal):
        """Detect motion artifacts"""
        # Compute derivative
        derivative = np.diff(ecg_signal)
        
        # Look for sudden changes
        threshold = 3 * np.std(derivative)
        sudden_changes = np.sum(np.abs(derivative) > threshold)
        
        quality = 1 - (sudden_changes / len(derivative))
        return max(0, min(1, quality))
    
    def _rpeak_sqi(self, ecg_signal):
        """Assess R-peak detection reliability"""
        from scipy.signal import find_peaks
        
        # Find peaks
        peaks, properties = find_peaks(ecg_signal, distance=self.fs//2, prominence=0.3)
        
        if len(peaks) < 3:
            return 0.3
        
        # Check RR interval consistency
        rr_intervals = np.diff(peaks) / self.fs
        rr_cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-6)
        
        # Normal RR intervals should have low CV (<0.2)
        quality = 1 - min(1, rr_cv)
        
        return quality
    
    def _powerline_sqi(self, ecg_signal):
        """Check for 50/60 Hz interference"""
        # Compute FFT
        fft = np.fft.fft(ecg_signal)
        freqs = np.fft.fftfreq(len(ecg_signal), 1/self.fs)
        
        # Look for power at 50/60 Hz
        powerline_freqs = [50, 60]
        total_power = np.sum(np.abs(fft))
        
        interference = 0
        for pl_freq in powerline_freqs:
            idx = np.argmin(np.abs(np.abs(freqs) - pl_freq))
            interference += np.abs(fft[idx])
        
        ratio = interference / (total_power + 1e-6)
        
        # Lower interference = better quality
        return max(0, min(1, 1 - 10*ratio))

# Create SQI dictionary for vital_sqi compatibility
def create_sqi_dict():
    """Create SQI dictionary for use with vital_sqi toolkit"""
    sqi_dict = {
        "skewness": {
            "name": "skewness",
            "def": [
                {"op": ">", "value": "0.5", "label": "reject"},
                {"op": "<", "value": "-0.5", "label": "reject"},
                {"op": ">=", "value": "-0.5", "label": "accept"},
                {"op": "<=", "value": "0.5", "label": "accept"}
            ]
        },
        "kurtosis": {
            "name": "kurtosis",
            "def": [
                {"op": ">", "value": "5", "label": "reject"},
                {"op": "<", "value": "2", "label": "reject"},
                {"op": ">=", "value": "2", "label": "accept"},
                {"op": "<=", "value": "5", "label": "accept"}
            ]
        },
        "entropy": {
            "name": "entropy",
            "def": [
                {"op": ">", "value": "0.8", "label": "accept"},
                {"op": "<=", "value": "0.8", "label": "reject"}
            ]
        }
    }
    
    with open('utils/sqi_dict.json', 'w') as f:
        json.dump(sqi_dict, f, indent=2)
    
    return sqi_dict

if __name__ == "__main__":
    # Test SQI
    sqi = SignalQualityIndex()
    
    # Generate test signal
    t = np.linspace(0, 10, 3600)
    clean_signal = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2.4*t)
    
    # Add some noise
    noisy_signal = clean_signal + 0.2*np.random.randn(len(t))
    
    # Compute SQIs
    clean_quality = sqi.compute_all_sqis(clean_signal)
    noisy_quality = sqi.compute_all_sqis(noisy_signal)
    
    print("Clean Signal Quality:", clean_quality['overall_quality'])
    print("Noisy Signal Quality:", noisy_quality['overall_quality'])