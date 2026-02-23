"""
Training Script
Copy this into train.py in your root folder
"""

import numpy as np
from utils.model import ArrhythmiaDetector

def generate_sample_data():
    """Generate sample ECG data for testing"""
    np.random.seed(42)
    
    # Generate synthetic ECG-like signals
    n_samples = 1000
    time = np.linspace(0, 10, 3600)  # 10 seconds at 360 Hz
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Create base signal
        t = np.linspace(0, 10, 3600)
        
        # Normal sinus rhythm
        if i % 5 == 0:
            signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
            label = 0
        # Atrial Fibrillation
        elif i % 5 == 1:
            signal = np.sin(2 * np.pi * np.random.uniform(1, 3) * t) + np.random.normal(0, 0.1, len(t))
            label = 1
        # Ventricular Tachycardia
        elif i % 5 == 2:
            signal = 1.5 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.05, len(t))
            label = 2
        # Supraventricular Tachycardia
        elif i % 5 == 3:
            signal = np.sin(2 * np.pi * 2.0 * t) + 0.3 * np.sin(2 * np.pi * 4.0 * t)
            label = 3
        # Bradycardia
        else:
            signal = np.sin(2 * np.pi * 0.8 * t) + 0.2 * np.sin(2 * np.pi * 1.6 * t)
            label = 4
        
        # Add some noise
        signal += np.random.normal(0, 0.05, len(t))
        
        X.append(signal)
        y.append(label)
    
    return np.array(X), np.array(y)

def main():
    print("=" * 50)
    print("Training Arrhythmia Detection Model")
    print("=" * 50)
    
    # Generate data
    print("\nGenerating sample data...")
    X, y = generate_sample_data()
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating model...")
    detector = ArrhythmiaDetector()
    detector.create_model(input_shape=(3600, 1), num_classes=5)
    
    # Train model
    print("\nTraining model...")
    history = detector.train(
        X_train.reshape(-1, 3600, 1),
        y_train,
        X_val.reshape(-1, 3600, 1),
        y_val,
        epochs=10
    )
    
    # Save model
    print("\nSaving model...")
    detector.save_model('models/arrhythmia_model.h5')
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()