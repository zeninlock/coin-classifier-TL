# 🪙 coin-classifier-TL

## 🧠 Coin Classification using Transfer Learning

This project demonstrates a deep learning pipeline for classifying coins into 315 unique classes using EfficientNetB0 as a feature extractor. It was built as a transfer learning project using TensorFlow/Keras, with a focus on handling class imbalance, using image augmentation, and optimizing validation accuracy.

---

## 📂 Dataset

The dataset consists of ~11,000 labeled coin images from 315 distinct classes. It includes a wide variety of coin types from different countries, which often look visually similar, adding to the classification challenge.

👉 **Download the dataset from Kaggle:**  
[https://www.kaggle.com/competitions/dl4cv-coin-classification/overview](https://www.kaggle.com/competitions/dl4cv-coin-classification/overview)

---

## 📌 Project Highlights

- **Model Architecture:** EfficientNetB0 (pretrained on ImageNet) + custom classification head  
- **Training Accuracy:** Up to 74%  
- **Validation Accuracy:** Consistently reached 73–75% depending on run  
- **Class Count:** 315  
- **Augmentation:** Random flips, rotations, zoom, contrast, and brightness  
- **Regularization:** EarlyStopping, Dropout, and class weighting  
- **Input Size:** 224x224 (as required by EfficientNetB0)  
- **Batch Size:** 64 (optimal balance between speed and performance)  
- **Evaluation:** Classification report + Confusion matrix  

---

## 🧠 Model Overview

- Used EfficientNetB0 as a frozen feature extractor  
- Custom top layers:  
  - `GlobalAveragePooling2D`  
  - `Dense(256)` + `Dropout(0.3)`  
  - `Dense(128)` + `Dropout(0.3)`  
  - Output: `Dense(315)` with `softmax`  

---

## 🚀 How to Run

1. Clone this repository or open the notebook in Google Colab  
2. Download and unzip the dataset from Kaggle. Then sort the images into 315 labels  
3. Run the notebook cells sequentially  

---

## 📊 Evaluation & Metrics

After training:

- **Classification Report:** Precision, recall, and F1-score calculated for all 315 classes  
- **Confusion Matrix:** Visualized to assess common misclassifications  

💡 Many misclassifications were observed among coins with similar designs from different countries (e.g., 1 Cent from USA vs Australia)

---

## 🧪 Training Strategy

- **Transfer Learning:** Used pretrained EfficientNetB0 without fine-tuning to save time and resources  
- **Data Augmentation:** Applied to prevent overfitting and improve generalization  
- **Early Stopping & Checkpointing:** Saved best model during training based on validation accuracy  
- **Class Weighting:** Handled class imbalance via `compute_class_weight`  

---

## 📈 Challenges & Learnings

### ✅ What Worked Well:
- EfficientNetB0 offered a good tradeoff between speed and accuracy  
- Augmentation significantly improved generalization  
- Class weighting helped reduce bias toward dominant classes  

### ⚠️ What Could Be Improved:
- Some visually similar coins were hard to differentiate  
- Model could benefit from fine-tuning EfficientNet  
- Dataset could be further cleaned to reduce noise  

---

## 🔮 Future Improvements

- Unfreeze EfficientNetB0 layers for fine-tuning  
- Explore heavier architectures (e.g., ResNet50) if training time allows  
- Improve preprocessing and label quality  
- Implement top-k accuracy metrics for practical evaluation  

---

## 💭 Reflections

This was one of my first major CNN projects, and I had to learn convolutional networks, image augmentation, and model training on the go. While challenging, the project gave me a deep understanding of model design, overfitting control, and evaluation strategies.

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Google Colab  

---

## 📧 Contact

For feedback or collaboration, feel free to reach out!
