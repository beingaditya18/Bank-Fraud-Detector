# 🏦 Bank Fraud Detection System

A powerful, real-time fraud detection system built with **Isolation Forest** and an interactive **Streamlit dashboard** to identify suspicious banking transactions.



## 🚀 Quick Start

```bash
git clone https://github.com/beingaditya18/Bank-Fraud-Detector.git
cd Bank-Fraud-Detector
pip install -r requirements.txt
python train_model.py   # One-time model training
streamlit run app.py    # Launch the dashboard
```

---

## 📚 About

Detect fraudulent transactions using machine learning on real-world credit card data.  
Features:

- ✅ **Isolation Forest** anomaly detection
- ✅ **Real-time** predictions with a beautiful Streamlit UI
- ✅ Clean, modular, production-ready code
- ✅ Easy-to-read analytics & results

---

## 🏗️ Project Structure

```
.
├── app.py                # Streamlit dashboard
├── train_model.py        # Model training script
├── data/                 # Place Kaggle creditcard.csv here
├── model/                # Saved model (.pkl) and scaler
├── utils/                # Preprocessing and prediction modules
├── assets/               # Images, banners
└── requirements.txt
```

---

## 🧠 Model

- **Algorithm**: Isolation Forest (unsupervised)
- **Contamination**: 0.0017
- **Features Scaled**: `Time`, `Amount`
- **Output**: `Fraud` / `Not Fraud`

---

## 📊 Results

| Metric       | Value   |
| ------------ | ------- |
| Accuracy     | 99.75%  |
| Fraud Recall | ~33%    |

---

## 📥 Dataset

- Download: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` in the `data/` folder.

---

## 🛠️ Next Steps

- Add supervised models (e.g., SVM)
- Store results in SQLite
- Advanced analytics & heatmaps



