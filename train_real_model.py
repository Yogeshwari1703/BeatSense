"""
Train CNN-LSTM on REAL MIT-BIH Arrhythmia Database
This matches the architecture from your paper
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

from data_loader import MITBIHLoader

class MITBIH_CNN_LSTM:
    """
    CNN-LSTM model trained on REAL MIT-BIH data
    Exactly as described in your paper
    """
    
    def __init__(self, input_shape=(3600, 1), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['Normal', 'SVE', 'Ventricular', 'Fusion', 'Other']
    
    def build_model(self):
        """
        Build the CNN-LSTM architecture from your paper
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # CNN layers for morphological features
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # LSTM layers for temporal dynamics
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(64),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model built successfully")
        print(model.summary())
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """
        Train the model on REAL MIT-BIH data
        """
        if self.model is None:
            self.build_model()
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('models/mitbih_cnn_lstm.h5', save_best_only=True)
        ]
        
        print("\n🚀 Training on REAL MIT-BIH data...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Epochs: {epochs}")
        print("="*60)
        
        # Train
        history = self.model.fit(
            X_train.reshape(-1, 3600, 1),
            y_train,
            validation_data=(X_val.reshape(-1, 3600, 1), y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        """
        print("\n📊 Evaluating on test data...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test.reshape(-1, 3600, 1))
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        accuracy = np.mean(y_pred == y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Get unique classes present in test data
        present_classes = np.unique(np.concatenate([y_test, y_pred]))
        present_class_names = [self.class_names[i] for i in present_classes]
        
        print(f"\nClasses present in test data: {present_class_names}")
        
        # Classification report (only for classes that exist)
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            labels=present_classes,
            target_names=present_class_names,
            zero_division=0
        ))
        
        # Confusion matrix (full)
        cm = confusion_matrix(y_test, y_pred, labels=range(self.num_classes))
        
        return y_pred, cm
    
    def plot_training_history(self, history):
        """
        Plot training curves
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        # Only show classes that have non-zero values
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        non_zero_rows = np.where(row_sums > 0)[0]
        non_zero_cols = np.where(col_sums > 0)[0]
        non_zero_indices = np.union1d(non_zero_rows, non_zero_cols)
        
        cm_filtered = cm[non_zero_indices][:, non_zero_indices]
        class_names_filtered = [self.class_names[i] for i in non_zero_indices]
        
        sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names_filtered, 
                   yticklabels=class_names_filtered)
        plt.title('Confusion Matrix (Active Classes Only)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()


# Main execution
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    print("="*60)
    print("REAL MIT-BIH ARRHYTHMIA DETECTION")
    print("Training CNN-LSTM on actual dataset")
    print("="*60)
    
    # Step 1: Load REAL data
    print("\n📥 Loading MIT-BIH data...")
    loader = MITBIHLoader()
    X, y = loader.load_all_data(max_records=15)  # Start with 15 records
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = loader.get_train_test_split(X, y, test_size=0.2)
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Final split:")
    print(f"   Train: {len(X_train)} segments")
    print(f"   Validation: {len(X_val)} segments")
    print(f"   Test: {len(X_test)} segments")
    
    # Step 3: Build model
    print("\n🔧 Building CNN-LSTM model...")
    model_trainer = MITBIH_CNN_LSTM()
    model_trainer.build_model()
    
    # Step 4: Train (this takes time - let it run)
    print("\n⏰ Starting training (this will take 2-3 hours for 30 epochs)...")
    history = model_trainer.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=30,
        batch_size=32
    )
    
    # Step 5: Evaluate
    print("\n📈 Evaluating model...")
    y_pred, cm = model_trainer.evaluate(X_test, y_test)
    
    # Step 6: Plot results
    model_trainer.plot_training_history(history)
    model_trainer.plot_confusion_matrix(cm)
    
    print("\n✅ REAL IMPLEMENTATION COMPLETE!")
    print("   Model saved to: models/mitbih_cnn_lstm.h5")
    print("   Training curves: training_history.png")
    print("   Confusion matrix: confusion_matrix.png")