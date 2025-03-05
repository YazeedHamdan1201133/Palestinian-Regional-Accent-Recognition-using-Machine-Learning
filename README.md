# 🗣️ Palestinian Regional Accent Recognition using Machine Learning

##  Overview
This project aims to **automatically recognize and classify Palestinian regional accents** from four major regions:  
- **Jerusalem**
- **Hebron**
- **Nablus**
- **Ramallah**

We developed and compared **three machine learning models**:
✔ **Support Vector Machine (SVM)**
✔ **Gaussian Mixture Model - Support Vector Machine (GMM-SVM)**
✔ **K-Nearest Neighbors (KNN)**

These models were trained on **acoustic features** extracted from speech recordings, including:
- **Mel-Frequency Cepstral Coefficients (MFCCs)**
- **Spectral Contrast**
- **Mel-Spectrogram**
- **Audio Energy**

---

## 🚀 Features
- **Accent Classification**: Predicts the regional Palestinian accent from speech.
- **Feature Extraction**: Uses advanced speech-processing techniques like MFCCs, contrast, and Mel-spectrogram.
- **Machine Learning Models**: Implements and compares **SVM, GMM-SVM, and KNN** for classification.
- **User Input Audio Prediction**: Allows users to test their own recordings.

---

## 📂 Files in this Repository
- **📄 Spoken_Paper.pdf** → Research paper describing methodology and results.
- **📜 SVM.py** → Implements **Support Vector Machine** for accent recognition.
- **📜 bestSVM.py** → Optimized **SVM model** with **Mel-spectrogram and Energy features**.
- **📜 KNN.py** → Implements **K-Nearest Neighbors (KNN)**.
- **📜 GmmSvm.py** → Implements **Gaussian Mixture Model - Support Vector Machine (GMM-SVM)**.

---

## 🏗️ Methodology

### **1️⃣ Data Preprocessing**
- Audio files were collected for **four Palestinian accents**.
- Files were **renamed and checked** for consistency.
- The dataset contained **10 speakers per accent**.

### **2️⃣ Feature Extraction**
We extracted **acoustic features** from each speech file:
- **MFCCs** → Captures pitch and tone variations.
- **Spectral Contrast** → Highlights harmonic content differences.
- **Mel-Spectrogram** → Provides a human-auditory-inspired frequency representation.
- **Audio Energy** → Measures the loudness of speech.

### **3️⃣ Model Training**
Three classification models were trained:
✔ **SVM** → Linear kernel for better separation.  
✔ **GMM-SVM** → Uses GMM probabilities as features for SVM.  
✔ **KNN** → Uses neighborhood classification based on similarity.

### **4️⃣ Model Evaluation**
Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

#### 📊 **Performance Summary**
| Model       | Accuracy | Precision | Recall | F1-Score |
|------------|----------|------------|--------|----------|
| **GMM-SVM** | 75% | 80% | 75% | 74% |
| **SVM (Best Version)** | 75% | 75% | 75% | 74% |
| **SVM (Baseline)** | 70% | 79% | 70% | 67% |
| **KNN** | 65% | 80% | 65% | 63% |

**🔍 Key Findings**:
✔ **GMM-SVM and Optimized SVM achieved the highest accuracy (75%)**.  
✔ **KNN performed the worst (65%) due to its sensitivity to small datasets**.  
✔ **Adding Spectral Contrast & Mel-Spectrogram improved SVM's accuracy**.

---

## 📎 How to Run the Code?

### 1️⃣ Install Required Libraries
Ensure you have all required dependencies installed. Run:

```sh
pip install numpy librosa scikit-learn matplotlib seaborn python_speech_features
```
### 2️⃣ Run Accent Recognition
Each model has its own script. Run any of the following:

```sh
python SVM.py
python bestSVM.py
python GmmSvm.py
python KNN.py
```
### 3️⃣ Predict an Accent
Each script provides an option to upload an audio file for prediction:
The script will extract features from the uploaded file.
It will predict the regional accent (Jerusalem, Hebron, Nablus, or Ramallah).
The predicted accent will be displayed

## 📢 Conclusion
This project successfully **classifies Palestinian accents** using **speech processing and machine learning**.

✔ **GMM-SVM and Best SVM performed best (75% accuracy)**.  
✔ **MFCC, Spectral Contrast, and Mel-Spectrogram significantly improved classification**.  
✔ **Larger datasets can improve results further**.  

---

## 🛠 Technologies Used
- **Python** 🐍  
- **Librosa** 🎵 (Feature Extraction)  
- **Scikit-learn** 🤖 (Machine Learning)  
- **Matplotlib & Seaborn** 📊 (Data Visualization)  
- **Tkinter** 🎤 (User File Selection for Accent Prediction)  

---

## 📫 Contact
For any questions or discussions, feel free to reach out:

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:yazedyazedl2020@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/yazeed-hamdan-59b83b281/)  
