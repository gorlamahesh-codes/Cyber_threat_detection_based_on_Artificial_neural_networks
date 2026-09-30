# 🛡️ CyberGuard AI: Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles

A desktop application built with **Python, Tkinter, scikit-learn and TensorFlow** that classifies cyber threats from security event logs. It converts log/event text into TF-IDF event profiles, trains multiple machine learning and deep learning models, compares their performance, and predicts the threat category of new events through an interactive security dashboard.

---

## 📌 Overview

Security systems produce huge volumes of event logs, and manually spotting malicious activity is slow and error-prone. This project treats each log entry as an **event profile**, vectorizes it with TF-IDF, and uses supervised learning to detect and categorize threats automatically.

The GUI wraps the whole pipeline (dataset loading, preprocessing, training, evaluation, visualization and prediction) in a single dashboard, so no coding is needed to run experiments.

## ✨ Features

- 🔐 **Administrator login** screen before accessing the console
- 📂 **Dataset upload** for CSV and Excel (`.csv`, `.xlsx`, `.xls`) files
- 🧹 **Automatic preprocessing**: merges all non-label columns into event documents, encodes labels
- 🔤 **TF-IDF vectorization** (up to 5,000 features, unigrams + bigrams)
- ✂️ **Event vector generation** with an 80/20 train/test split (stratified when possible)
- 🤖 **Multiple classifiers** to compare:
  - Support Vector Machine (SVM, linear kernel)
  - Artificial Neural Network (ANN, dense layers with dropout)
  - Long Short-Term Memory (LSTM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Bernoulli)
  - Decision Tree and Random Forest (implemented in the code)
- 📊 **Performance metrics**: Accuracy, Precision, Recall and F1-score (macro averaged)
- 📈 **Comparison graphs** using Matplotlib
- 🔎 **Live threat prediction**: type an event message and get the predicted threat category from the most recently trained model
- 🖥️ **Modern dashboard UI** with status cards, a live security console, and a control center

## 🧠 How It Works

```
Dataset (CSV/Excel)
        │
        ▼
Label encoding + event text creation
        │
        ▼
TF-IDF vectorization  ──►  Event vectors
        │
        ▼
Train/Test split (80/20)
        │
        ▼
Train model (SVM / ANN / LSTM / KNN / NB ...)
        │
        ▼
Evaluate (Accuracy, Precision, Recall, F1)
        │
        ▼
Predict threat category for new events
```

## 🗂️ Dataset Format

Your dataset must contain a **`labels`** column holding the threat category. The following alternative column names are automatically renamed to `labels`: `label`, `Label`, `Labels`, `threat`, `Threat`, `class`, `Class`.

All remaining columns are joined into a single text "event document" for each row.

Example:

| source_ip     | event_description                | labels        |
|---------------|----------------------------------|---------------|
| 192.168.1.10  | Multiple failed login attempts   | Brute Force   |
| 10.0.0.5      | Unusual outbound data transfer   | Data Exfil    |
| 172.16.0.8    | Normal user login                | Normal        |

## 🧰 Tech Stack

| Area              | Tools                              |
|-------------------|------------------------------------|
| Language          | Python 3.8+                        |
| GUI               | Tkinter, ttk, ScrolledText         |
| Data handling     | NumPy, Pandas                      |
| Machine learning  | scikit-learn                       |
| Deep learning     | TensorFlow / Keras (optional)      |
| Visualization     | Matplotlib                         |

## ⚙️ Installation

1. **Clone the repository**

   ```bash
  https://github.com/gorlamahesh-codes/Cyber_threat_detection_based_on_Artificial_neural_networks.git
   cd <your-repo-name>
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install numpy pandas scikit-learn matplotlib openpyxl
   pip install tensorflow        # optional, required for ANN and LSTM
   ```

   > If TensorFlow is not installed, the app still opens and all scikit-learn models work. ANN and LSTM will show an install message instead.

## ▶️ Usage

```bash
python main.py
```

1. **Log in** with the administrator credentials (see the note below).
2. Click **Upload Dataset** and choose your CSV/Excel file.
3. Click **Run TF-IDF** to extract features.
4. Click **Generate Event Vector** to prepare training and testing data.
5. Run any model (**SVM, ANN, LSTM, KNN, Naive Bayes**).
6. Use **Show Results**, **Accuracy Graph**, or **Precision / Recall / F1** to compare models.
7. Enter an event message in the **Threat Event Input** box and click **Predict Threat**.

> 🔑 **Demo credentials:** `mahesh` / `mahesh`. These are hard-coded for demonstration only. Change them in `authenticate_user()` before any real use.

## 📊 Output

- Metrics for each trained model are printed in the built-in security console.
- A results table compares all trained algorithms side by side.
- Bar charts visualize the models' performance.
- Predictions display the input event, the active model and the predicted threat label.

## 📁 Project Structure

```
├── main.py        # Complete application (UI, preprocessing, models, prediction)
├── README.md      # Project documentation
└── dataset/       # Your CSV/Excel event log datasets (not included)
```

## 🚀 Future Improvements

- Add Decision Tree and Random Forest buttons to the dashboard
- Show a confusion matrix and per-class metrics
- Save and load trained models (`joblib` / Keras `.h5`)
- Real-time log monitoring and alerts
- Replace hard-coded login with secure, hashed credential storage
- Hyperparameter tuning and cross-validation

## 📜 License

This project is released under the [MIT License](LICENSE). Feel free to use, modify and distribute it.

## 👤 Author

**Mahesh**
Feel free to open an issue or submit a pull request for suggestions and improvements.

---

⭐ If you found this project useful, consider giving it a star!
