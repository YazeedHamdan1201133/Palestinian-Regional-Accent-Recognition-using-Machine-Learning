import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

# Define paths to the training and testing data
train_data_path = r'C:\Users\hp\Desktop\PS_Accents'
test_data_path = r'C:\Users\hp\Desktop\testingdata'

# Function to extract features from an audio file using various features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)

        # Combine all features into a single feature vector
        features = np.hstack((mfccs_scaled, spectral_contrast_scaled))
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(e)
        return None
    return features

# Function to load data from a directory and extract features
def load_data(data_path):
    X, y = [], []
    accents = ['Jerusalem', 'Nablus', 'Hebron', 'Ramallah']
    for accent in accents:
        accent_path = os.path.join(data_path, accent)
        if not os.path.exists(accent_path):
            print(f"Directory not found: {accent_path}")
            continue
        print(f"Accessing directory: {accent_path}")
        for filename in os.listdir(accent_path):
            file_path = os.path.join(accent_path, filename)
            print(f"Processing file: {file_path}")
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(accent)
            else:
                print(f"Failed to extract features from: {file_path}")
    if not X:
        print("No data was loaded. Please check the directories and file contents.")
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) if y else []
    return np.array(X), y_encoded, le

# Load training and testing data
print("Loading and processing training data...")
X_train, y_train, label_encoder = load_data(train_data_path)
print("Loading and processing testing data...")
X_test, y_test, _ = load_data(test_data_path)

if X_train.size == 0 or X_test.size == 0:
    print("Training or testing data is empty. Cannot proceed with training the model.")
else:
    # Initialize and train the Gaussian Mixture Model
    print("Training the Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=4, covariance_type='diag')
    gmm.fit(X_train)

    # Extract GMM posteriors
    X_train_gmm = gmm.predict_proba(X_train)
    X_test_gmm = gmm.predict_proba(X_test)

    # Combine original features with GMM posteriors
    X_train_combined = np.hstack((X_train, X_train_gmm))
    X_test_combined = np.hstack((X_test, X_test_gmm))

    # Initialize and train the Support Vector Machine
    print("Training the Support Vector Machine with GMM posteriors...")
    svm = SVC(kernel='linear')  # Using a linear kernel; you can try 'rbf', 'poly', etc.
    svm.fit(X_train_combined, y_train)

    # Evaluate the model on the testing data
    print("Evaluating the model...")
    y_pred = svm.predict(X_test_combined)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of SVM with GMM posteriors on test data: {accuracy:.2f}')

    # Print the classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

    # Compute and plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix "GMM-SVM"')
    plt.show()

# Function to open a file dialog to select an audio file
def file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hides the small tkinter window
    file_path = filedialog.askopenfilename()
    return file_path

# Function to predict the accent of a given audio file
def predict_accent():
    file_path = file_dialog()
    if file_path:
        print(f"Predicting accent for file: {file_path}")
        features = extract_features(file_path)
        if features is not None:
            features = np.expand_dims(features, axis=0)  # Reshape for a single sample
            gmm_posteriors = gmm.predict_proba(features)
            features_combined = np.hstack((features, gmm_posteriors))
            predicted_class = svm.predict(features_combined)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            return predicted_label
        else:
            return "Could not extract features and predict the accent."
    else:
        return "No file selected."

# Example usage
predicted_accent = predict_accent()
print(f"The predicted accent is: {predicted_accent}")
