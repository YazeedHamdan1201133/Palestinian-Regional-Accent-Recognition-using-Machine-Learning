import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

def read_wav_file(filename):
    """Read a WAV file and return its sampling frequency and audio data."""
    try:
        sampling_freq, audio = wav.read(filename)
        if audio.size == 0:
            raise ValueError("Audio file is empty")
        return sampling_freq, audio
    except Exception as e:
        raise IOError(f"Could not read file {filename}: {e}")

def extract_features(directory):
    """Extract features from all WAV files in the given directory."""
    features = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.wav'):
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")
                try:
                    sampling_freq, audio = read_wav_file(filepath)
                    mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
                    if mfcc_features.size > 0:
                        features.append(np.mean(mfcc_features, axis=0))
                        labels.append(os.path.basename(root))
                    else:
                        print(f"Warning: No MFCC features extracted for {filename}")
                except Exception as e:
                    print(f"Error: {e}")
    return features, labels

def train_knn(features, labels, num_neighbors=8):
    """Train KNN model."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Scale features for KNN
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_model.fit(features_scaled, labels)
    return knn_model, scaler

def predict(model, scaler, features):
    """Predict using the trained KNN model."""
    features_scaled = scaler.transform(features)  # Scale features using the same scaler as training
    return model.predict(features_scaled)

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix "KNN"')
    plt.show()

def evaluate_models(train_path, test_path):
    """Train models on training data and evaluate them on test data."""
    print("Loading training data...")
    train_features, train_labels = extract_features(train_path)
    print("Loading testing data...")
    test_features, test_labels = extract_features(test_path)

    if not train_features or not test_features:
        print("Insufficient data for training or testing.")
        return

    print("Training KNN model...")
    knn_model, scaler = train_knn(train_features, train_labels)
    print("Predicting test data...")
    test_predictions = predict(knn_model, scaler, test_features)

    if test_predictions.size > 0:
        accuracy = accuracy_score(test_labels, test_predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("Classification Report:")
        print(classification_report(test_labels, test_predictions))

        cm = confusion_matrix(test_labels, test_predictions)
        plot_confusion_matrix(cm, classes=np.unique(test_labels))
    else:
        print("No predictions made.")

    return knn_model, scaler

def file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hides the small tkinter window
    return filedialog.askopenfilename()

def predict_file(knn_model, scaler):
    """Predict the label of a single WAV file selected by the user."""
    filepath = file_dialog()
    if filepath:
        try:
            sampling_freq, audio = read_wav_file(filepath)
            mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
            if mfcc_features.size > 0:
                features = np.mean(mfcc_features, axis=0).reshape(1, -1)
                prediction = predict(knn_model, scaler, features)
                print(f"Predicted label for {filepath}: {prediction[0]}")
            else:
                print(f"No MFCC features extracted for {filepath}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    train_data_path = r'C:\Users\hp\Desktop\PS_Accents'
    test_data_path = r'C:\Users\hp\Desktop\testingdata'
    knn_model, scaler = evaluate_models(train_data_path, test_data_path)
    predict_file(knn_model, scaler)
