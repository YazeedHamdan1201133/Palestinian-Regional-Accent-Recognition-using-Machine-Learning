import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Define paths to the training and testing data
train_data_path = r'C:\Users\hp\Desktop\PS_Accents'
test_data_path = r'C:\Users\hp\Desktop\testingdata'

# Function to extract features from an audio file using MFCC, Mel-spectrogram, and audio energy
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features = np.hstack((mfccs_scaled))

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
    # Initialize and train the Support Vector Machine
    print("Training the Support Vector Machine...")
    svm = SVC(kernel='linear')  # Using a linear kernel
    svm.fit(X_train, y_train)

    # Evaluate the model on the testing data
    print("Evaluating the model...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of SVM on test data: {accuracy:.2f}')
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix "SVM1"')
    plt.show()

    # Generate and print the classification report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:\n")
    print(class_report)

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
            predicted_class = svm.predict(features)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            return predicted_label
        else:
            return "Could not extract features and predict the accent."
    else:
        return "No file selected."

# Example usage
predicted_accent = predict_accent()
print(f"The predicted accent is: {predicted_accent}")
