# ML-TASK-2
# 🛡️ Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using anomaly detection techniques such as **Isolation Forest** and **Autoencoders**. The dataset is highly imbalanced, and models are evaluated based on their ability to correctly identify rare fraud cases.

## 🎯 Objective

Identify potentially fraudulent transactions using machine learning, allowing financial institutions to reduce losses and take proactive measures.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**:
  - `Time`: Seconds elapsed between each transaction
  - `Amount`: Transaction amount (standardized)
  - `V1` to `V28`: PCA-transformed features
  - `Class`: 1 (fraud), 0 (normal)

---

## 🔍 Project Structure

- `creditcard.csv`: Dataset
- `fraud_detection.ipynb`: Google Colab-compatible notebook with full implementation
- `README.md`: Project documentation

---

## 🧪 Models Used

### 1. ✅ Isolation Forest
- Anomaly detection algorithm
- Trained on full dataset (unsupervised)
- Predicts outliers as potential frauds

### 2. ✅ Autoencoder (Neural Network)
- Trained only on non-fraudulent transactions
- Reconstructs normal data
- High reconstruction error signals anomalies

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras
- Matplotlib, Seaborn

---

## ⚙️ How to Run

1. Open `fraud_detection.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `creditcard.csv`
3. Run each cell in order
4. View confusion matrix, classification report, and ROC-AUC score for both models

---

## 📊 Evaluation Metrics

- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC AUC Score**
- **Reconstruction Error Visualization** (Autoencoder)

---

## 📌 Key Insights

- The dataset is **highly imbalanced** — only ~0.17% of transactions are fraud.
- **Autoencoders** perform well when trained on only normal data.
- **Isolation Forest** can be used without labels (unsupervised).
- Visualizing reconstruction error helps set a clear anomaly threshold.

---

## 🧠 Future Improvements

- Use **SHAP or LIME** for model interpretability
- Deploy a **real-time monitoring system**
- Add **feedback loops** to retrain on newly discovered fraud patterns
- Explore **ensemble anomaly detectors**

---

## 🤝 Acknowledgements

- ULB Machine Learning Group for the dataset
- Google Colab for cloud-based development

---

