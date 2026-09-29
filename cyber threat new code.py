import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

# TensorFlow is optional so the application can still open if it is not installed.
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


# ============================================================
# APPLICATION DATA
# ============================================================

APP_TITLE = "Cyber Threat Detection | AI Security Analytics"

COLORS = {
    "navy": "#071A3A",
    "navy2": "#0B2A57",
    "navy3": "#0D376E",
    "blue": "#1688FF",
    "blue2": "#35A7FF",
    "cyan": "#16C7E8",
    "green": "#16B887",
    "green2": "#20D39B",
    "purple": "#7347F5",
    "purple2": "#8A5CFF",
    "red": "#F0445D",
    "orange": "#F58220",
    "yellow": "#F5B83D",
    "pink": "#EF3E8D",
    "bg": "#F4F7FC",
    "white": "#FFFFFF",
    "text": "#17315C",
    "muted": "#7184A5",
    "border": "#DCE5F2",
    "console": "#061A2E",
    "console2": "#092640",
    "console_text": "#5CF2BE",
    "console_muted": "#8DB7D8",
}

root = tk.Tk()
root.title(APP_TITLE)
root.geometry("1450x900")
root.minsize(1200, 760)
root.configure(bg=COLORS["bg"])


# ============================================================
# GLOBAL STATE
# ============================================================

dataset = None
filename = ""
documents = []

X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None

tfidf_vectorizer = None
label_encoder = LabelEncoder()

trained_model = None
trained_model_type = None
results = {}

model_status = "Ready"


# ============================================================
# GENERAL HELPERS
# ============================================================

def update_clock():
    now = datetime.now()
    date_var.set(now.strftime("%a, %d %b %Y"))
    time_var.set(now.strftime("%I:%M %p"))
    root.after(1000, update_clock)


def print_output(message):
    output_text.configure(state="normal")
    output_text.insert(tk.END, str(message) + "\n")
    output_text.see(tk.END)
    output_text.configure(state="disabled")


def clear_console():
    output_text.configure(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.configure(state="disabled")


def set_model_status(value):
    global model_status
    model_status = value
    model_status_var.set(value)


def clear_results():
    results.clear()


def check_training_data():
    if X_train is None or X_test is None:
        print_output("Please upload a dataset, run TF-IDF and generate event vectors first.")
        return False
    return True


def set_last_model(model, model_type):
    global trained_model, trained_model_type
    trained_model = model
    trained_model_type = model_type
    set_model_status(f"{model_type} Ready")


def calculate_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    pre = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    results[name] = {
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
    }

    print_output("")
    print_output("=" * 62)
    print_output(f"{name} RESULTS")
    print_output("=" * 62)
    print_output(f"Accuracy : {acc:.2f}%")
    print_output(f"Precision: {pre:.2f}%")
    print_output(f"Recall   : {rec:.2f}%")
    print_output(f"F1 Score : {f1:.2f}%")
    print_output("=" * 62)


# ============================================================
# DATASET / TF-IDF
# ============================================================

def upload_dataset():
    global dataset, X, Y, documents, filename
    global X_train, X_test, y_train, y_test
    global tfidf_vectorizer, label_encoder

    path = filedialog.askopenfilename(
        title="Select Cyber Threat Dataset",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*"),
        ],
    )

    if not path:
        return

    try:
        if path.lower().endswith(".csv"):
            dataset = pd.read_csv(path)
        else:
            dataset = pd.read_excel(path)

        if "labels" not in dataset.columns:
            # Also accept common alternatives.
            alternatives = ["label", "Label", "Labels", "threat", "Threat", "class", "Class"]
            found = next((c for c in alternatives if c in dataset.columns), None)
            if found:
                dataset = dataset.rename(columns={found: "labels"})
            else:
                messagebox.showerror(
                    "Dataset Error",
                    "Your dataset must contain a 'labels' column."
                )
                dataset = None
                return

        dataset = dataset.dropna(how="all").copy()

        if dataset.empty:
            messagebox.showerror("Dataset Error", "The selected dataset is empty.")
            return

        # Encode target labels.
        dataset["labels"] = dataset["labels"].astype(str)
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(dataset["labels"])

        # Use every non-label column as event information.
        feature_df = dataset.drop(columns=["labels"]).fillna("")
        X_raw = feature_df.astype(str)

        documents = X_raw.apply(
            lambda row: " ".join(row.values),
            axis=1
        ).tolist()

        X = None
        X_train = X_test = y_train = y_test = None
        tfidf_vectorizer = None

        clear_results()
        set_model_status("Dataset Loaded")

        clear_console()
        print_output("Administrator authenticated successfully.")
        print_output("Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")
        print_output("Security Console is ready.")
        print_output("")
        print_output("DATASET LOADED SUCCESSFULLY")
        print_output("-" * 62)
        print_output(f"File    : {path}")
        print_output(f"Rows    : {len(dataset)}")
        print_output(f"Columns : {dataset.shape[1]}")
        print_output(f"Classes : {len(np.unique(Y))}")
        print_output(f"Labels  : {', '.join(map(str, label_encoder.classes_))}")

        filename_var.set(path.split("/")[-1].split("\\")[-1])
        dataset_var.set("Loaded")
        dataset_detail_var.set(f"{len(dataset):,} rows • {len(np.unique(Y))} classes")

    except Exception as exc:
        messagebox.showerror("Dataset Error", str(exc))


def tfidf_preprocessing():
    global X, tfidf_vectorizer

    if dataset is None:
        print_output("Please upload a dataset first.")
        return

    if not documents:
        print_output("No event documents were found.")
        return

    try:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True
        )

        X = tfidf_vectorizer.fit_transform(documents).toarray()

        clear_console()
        print_output("TF-IDF PREPROCESSING COMPLETED")
        print_output("-" * 62)
        print_output(f"Documents       : {len(documents)}")
        print_output(f"Feature Size    : {X.shape[1]}")
        print_output("Vectorization   : TF-IDF")
        print_output("Status          : Ready for event profiling")

        set_model_status("Features Ready")

    except Exception as exc:
        messagebox.showerror("TF-IDF Error", str(exc))


def create_event_vectors():
    global X_train, X_test, y_train, y_test

    if X is None or Y is None:
        print_output("Please upload the dataset and run TF-IDF first.")
        return

    if len(np.unique(Y)) < 2:
        print_output("At least two classes are required for classification.")
        return

    try:
        counts = pd.Series(Y).value_counts()

        if counts.min() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                test_size=0.20,
                random_state=42,
                shuffle=True,
                stratify=Y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                test_size=0.20,
                random_state=42,
                shuffle=True,
            )

        clear_console()
        print_output("EVENT VECTOR PROFILE GENERATED")
        print_output("-" * 62)
        print_output(f"Training Samples : {len(X_train)}")
        print_output(f"Testing Samples  : {len(X_test)}")
        print_output(f"Vector Dimension : {X.shape[1]}")
        print_output("Status           : Training data prepared")

        set_model_status("Event Vectors Ready")

    except Exception as exc:
        messagebox.showerror("Event Vector Error", str(exc))


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

def run_svm():
    global trained_model

    if not check_training_data():
        return

    try:
        clear_console()
        print_output("Training LSTM Model...")
        model = SVC(kernel="linear", probability=True, random_state=42)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        set_last_model(model, "SVM")
        calculate_metrics("LSTM", y_test, prediction)
        print_output("LSTM model is now active for Predict Threat.")

    except Exception as exc:
        messagebox.showerror("SVM Error", str(exc))


def run_ann():
    global trained_model

    if not check_training_data():
        return

    if not TENSORFLOW_AVAILABLE:
        print_output("TensorFlow is not installed. ANN cannot run.")
        print_output("Install it using: pip install tensorflow")
        return

    try:
        clear_console()
        print_output("Training ANN Model...")

        classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, classes)

        model = Sequential([
            Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.30),
            Dense(128, activation="relu"),
            Dropout(0.30),
            Dense(64, activation="relu"),
            Dense(classes, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X_train,
            y_train_cat,
            epochs=5,
            batch_size=32,
            verbose=0,
        )

        prediction = np.argmax(model.predict(X_test, verbose=0), axis=1)

        set_last_model(model, "ANN")
        calculate_metrics("ANN", y_test, prediction)
        print_output("ANN model is now active for Predict Threat.")

    except Exception as exc:
        messagebox.showerror("ANN Error", str(exc))


def run_lstm():
    global trained_model

    if not check_training_data():
        return

    if not TENSORFLOW_AVAILABLE:
        print_output("TensorFlow is not installed. LSTM cannot run.")
        print_output("Install it using: pip install tensorflow")
        return

    try:
        clear_console()
        print_output("Training SVM Model...")

        classes = len(np.unique(y_train))

        y_train_cat = to_categorical(y_train, classes)

        X_train_lstm = X_train.reshape(
            X_train.shape[0],
            X_train.shape[1],
            1,
        )

        X_test_lstm = X_test.reshape(
            X_test.shape[0],
            X_test.shape[1],
            1,
        )

        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dropout(0.30),
            Dense(64, activation="relu"),
            Dense(classes, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X_train_lstm,
            y_train_cat,
            epochs=3,
            batch_size=32,
            verbose=0,
        )

        prediction = np.argmax(
            model.predict(X_test_lstm, verbose=0),
            axis=1
        )

        set_last_model(model, "LSTM")
        calculate_metrics("SVM", y_test, prediction)
        print_output("SVM model is now active for Predict Threat.")

    except Exception as exc:
        messagebox.showerror("LSTM Error", str(exc))


def run_knn():
    if not check_training_data():
        return

    try:
        clear_console()
        print_output("Training K-Nearest Neighbors...")

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        set_last_model(model, "KNN")
        calculate_metrics("KNN", y_test, prediction)
        print_output("KNN model is now active for Predict Threat.")

    except Exception as exc:
        messagebox.showerror("KNN Error", str(exc))


def run_decision_tree():
    if not check_training_data():
        return

    try:
        clear_console()
        print_output("Training Decision Tree...")

        model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        set_last_model(model, "Decision Tree")
        calculate_metrics("Decision Tree", y_test, prediction)

    except Exception as exc:
        messagebox.showerror("Decision Tree Error", str(exc))


def run_random_forest():
    if not check_training_data():
        return

    try:
        clear_console()
        print_output("Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        set_last_model(model, "Random Forest")
        calculate_metrics("Random Forest", y_test, prediction)

    except Exception as exc:
        messagebox.showerror("Random Forest Error", str(exc))


def run_naive_bayes():
    if not check_training_data():
        return

    try:
        clear_console()
        print_output("Training Naive Bayes...")

        model = BernoulliNB()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        set_last_model(model, "Naive Bayes")
        calculate_metrics("Naive Bayes", y_test, prediction)
        print_output("Naive Bayes model is now active for Predict Threat.")

    except Exception as exc:
        messagebox.showerror("Naive Bayes Error", str(exc))


# ============================================================
# THREAT PREDICTION
# ============================================================

def predict_event():
    if trained_model is None:
        print_output("Please train at least one model first.")
        return

    if tfidf_vectorizer is None:
        print_output("Please run TF-IDF preprocessing first.")
        return

    text = input_entry.get().strip()

    if not text:
        print_output("Enter an event message before prediction.")
        input_entry.focus_set()
        return

    try:
        feature = tfidf_vectorizer.transform([text]).toarray()

        if trained_model_type == "LSTM":
            feature_model = feature.reshape(
                feature.shape[0],
                feature.shape[1],
                1
            )
            prediction = trained_model.predict(
                feature_model,
                verbose=0
            )
            predicted_id = int(np.argmax(prediction, axis=1)[0])

        elif trained_model_type == "ANN":
            prediction = trained_model.predict(feature, verbose=0)
            predicted_id = int(np.argmax(prediction, axis=1)[0])

        else:
            predicted_id = int(trained_model.predict(feature)[0])

        label = label_encoder.inverse_transform([predicted_id])[0]

        clear_console()
        print_output("THREAT PREDICTION COMPLETED")
        print_output("=" * 62)
        print_output(f"Active Model : {trained_model_type}")
        print_output("")
        print_output("Input Event:")
        print_output(text)
        print_output("")
        print_output("Predicted Threat:")
        print_output(f">>> {label}")
        print_output("")
        print_output("Prediction status: CLASSIFIED")

    except Exception as exc:
        messagebox.showerror("Prediction Error", str(exc))


# ============================================================
# RESULTS / GRAPHS
# ============================================================

def show_all_results():
    clear_console()

    if not results:
        print_output("No algorithms have been executed yet.")
        return

    print_output("MODEL PERFORMANCE RESULTS")
    print_output("=" * 78)
    print_output(
        f"{'Algorithm':<20}"
        f"{'Accuracy':>13}"
        f"{'Precision':>13}"
        f"{'Recall':>13}"
        f"{'F1':>13}"
    )
    print_output("-" * 78)

    for algorithm, metric in results.items():
        print_output(
            f"{algorithm:<20}"
            f"{metric['Accuracy']:>12.2f}%"
            f"{metric['Precision']:>12.2f}%"
            f"{metric['Recall']:>12.2f}%"
            f"{metric['F1']:>12.2f}%"
        )


def draw_graph(metric):
    if not results:
        print_output("Please run at least one algorithm first.")
        return

    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = [results[name][metric] for name in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.title(f"{metric} Comparison")
    plt.xlabel("Algorithms")
    plt.ylabel(f"{metric} (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


# ============================================================
# UI COMPONENT HELPERS
# ============================================================

def make_button(parent, text, command, bg, icon="", width=17):
    box = tk.Frame(parent, bg=bg, cursor="hand2")
    box.configure(width=width * 8, height=58)
    box.pack_propagate(False)

    icon_label = tk.Label(
        box,
        text=icon,
        bg=bg,
        fg="white",
        font=("Segoe UI Symbol", 19, "bold")
    )
    icon_label.pack(side="left", padx=(12, 7))

    text_box = tk.Frame(box, bg=bg)
    text_box.pack(side="left", fill="both", expand=True)

    title = tk.Label(
        text_box,
        text=text,
        bg=bg,
        fg="white",
        font=("Segoe UI", 10, "bold"),
        anchor="w"
    )
    title.pack(anchor="w", pady=(8, 0))

    subtitle_map = {
        "Upload Dataset": "CSV / Excel Files",
        "Run TF-IDF": "Extract Features",
        "Generate Event Vector": "Profile Events",
        "Run SVM": "Support Vector Machine",
        "Run ANN": "Neural Network",
        "Run LSTM": "Deep Learning",
        "Run KNN": "K-Nearest Neighbors",
        "Naive Bayes": "Classification",
        "Accuracy Graph": "Model Performance",
        "Precision / Recall / F1": "Detailed Metrics",
        "Show Results": "View Predictions",
    }

    subtitle = tk.Label(
        text_box,
        text=subtitle_map.get(text, ""),
        bg=bg,
        fg="#E4F0FF",
        font=("Segoe UI", 8),
        anchor="w"
    )
    subtitle.pack(anchor="w")

    def click(event=None):
        command()

    def enter(event=None):
        box.configure(bg=COLORS["blue2"])
        icon_label.configure(bg=COLORS["blue2"])
        text_box.configure(bg=COLORS["blue2"])
        title.configure(bg=COLORS["blue2"])
        subtitle.configure(bg=COLORS["blue2"])

    def leave(event=None):
        box.configure(bg=bg)
        icon_label.configure(bg=bg)
        text_box.configure(bg=bg)
        title.configure(bg=bg)
        subtitle.configure(bg=bg)

    for widget in (box, icon_label, text_box, title, subtitle):
        widget.bind("<Button-1>", click)
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    return box


def make_card(parent, title, value_var, detail_var, icon, accent):
    card = tk.Frame(
        parent,
        bg="white",
        highlightbackground=COLORS["border"],
        highlightthickness=1
    )

    icon_box = tk.Frame(card, bg=accent, width=58, height=58)
    icon_box.pack(side="left", padx=14, pady=12)
    icon_box.pack_propagate(False)

    tk.Label(
        icon_box,
        text=icon,
        bg=accent,
        fg="white",
        font=("Segoe UI Symbol", 23, "bold")
    ).pack(expand=True)

    info = tk.Frame(card, bg="white")
    info.pack(side="left", fill="both", expand=True, pady=9)

    tk.Label(
        info,
        text=title,
        bg="white",
        fg=COLORS["text"],
        font=("Segoe UI", 9)
    ).pack(anchor="w")

    tk.Label(
        info,
        textvariable=value_var,
        bg="white",
        fg="#102B55",
        font=("Segoe UI", 14, "bold")
    ).pack(anchor="w")

    tk.Label(
        info,
        textvariable=detail_var,
        bg="white",
        fg=COLORS["muted"],
        font=("Segoe UI", 8)
    ).pack(anchor="w")

    return card


def sidebar_item(parent, icon, text, active=False, command=None):
    bg = "#1588F8" if active else COLORS["navy"]

    item = tk.Frame(parent, bg=bg, height=42, cursor="hand2")
    item.pack(fill="x", padx=12, pady=3)
    item.pack_propagate(False)

    tk.Label(
        item,
        text=icon,
        bg=bg,
        fg="white",
        font=("Segoe UI Symbol", 16)
    ).pack(side="left", padx=(13, 13))

    tk.Label(
        item,
        text=text,
        bg=bg,
        fg="white",
        font=("Segoe UI", 10, "bold" if active else "normal")
    ).pack(side="left")

    if command:
        item.bind("<Button-1>", lambda e: command())
    return item


# ============================================================
# LOGIN
# ============================================================

def show_login():
    login = tk.Toplevel(root)
    login.title("Administrator Access")
    login.geometry("980x700")
    login.resizable(False, False)
    login.configure(bg="#07111F")

# Make login window visible and bring it to the front
    login.deiconify()
    login.lift()
    login.focus_force()
    login.attributes("-topmost", True)
    login.after(200, lambda: login.attributes("-topmost", False))

    left = tk.Frame(login, bg="#071A2F", width=490)
    left.pack(side="left", fill="both")
    left.pack_propagate(False)

    tk.Label(
        left,
        text="◈",
        bg="#071A2F",
        fg="#35A7FF",
        font=("Segoe UI Symbol", 44, "bold")
    ).pack(anchor="w", padx=52, pady=(70, 5))

    tk.Label(
        left,
        text="CyberGuard AI",
        bg="#071A2F",
        fg="white",
        font=("Segoe UI", 29, "bold")
    ).pack(anchor="w", padx=52)

    tk.Label(
        left,
        text="INTELLIGENT THREAT ANALYTICS",
        bg="#071A2F",
        fg="#35C8FF",
        font=("Segoe UI", 10, "bold")
    ).pack(anchor="w", padx=54, pady=(8, 28))

    tk.Label(
        left,
        text="Advanced event-profile analysis powered by\nmachine learning and neural network models.",
        justify="left",
        bg="#071A2F",
        fg="#A9BCD6",
        font=("Segoe UI", 11)
    ).pack(anchor="w", padx=54)

    for feature in [
        "●  Multi-model threat classification",
        "●  Event profile intelligence",
        "●  Performance analytics dashboard",
    ]:
        tk.Label(
            left,
            text=feature,
            bg="#071A2F",
            fg="#D9E7F7",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=54, pady=6)

    right = tk.Frame(login, bg="#F7FAFC")
    right.pack(side="right", fill="both", expand=True)

    form = tk.Frame(right, bg="#F7FAFC")
    form.pack(fill="both", expand=True, padx=65, pady=58)

    tk.Label(
        form,
        text="ADMIN PORTAL",
        bg="#F7FAFC",
        fg="#2563EB",
        font=("Segoe UI", 9, "bold")
    ).pack(anchor="w")

    tk.Label(
        form,
        text="Welcome back",
        bg="#F7FAFC",
        fg="#0F172A",
        font=("Segoe UI", 27, "bold")
    ).pack(anchor="w", pady=(8, 5))

    tk.Label(
        form,
        text="Sign in to continue to the security console.",
        bg="#F7FAFC",
        fg="#64748B",
        font=("Segoe UI", 10)
    ).pack(anchor="w", pady=(0, 28))

    tk.Label(
        form,
        text="USERNAME",
        bg="#F7FAFC",
        fg="#334155",
        font=("Segoe UI", 9, "bold")
    ).pack(anchor="w", pady=(0, 7))

    username = tk.Entry(
        form,
        font=("Segoe UI", 12),
        bg="white",
        fg="#0F172A",
        relief="solid",
        bd=1
    )
    username.pack(fill="x", ipady=10)

    tk.Label(
        form,
        text="PASSWORD",
        bg="#F7FAFC",
        fg="#334155",
        font=("Segoe UI", 9, "bold")
    ).pack(anchor="w", pady=(18, 7))

    password = tk.Entry(
        form,
        show="●",
        font=("Segoe UI", 12),
        bg="white",
        fg="#0F172A",
        relief="solid",
        bd=1
    )
    password.pack(fill="x", ipady=10)

    def login_window_close():
        login.destroy()
        root.deiconify()
        root.lift()
        print_output("Administrator authenticated successfully.")
        print_output("Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")
        print_output("Security Console is ready.")
        print_output("Upload a dataset to begin analysis...")

    def authenticate_user():
        if username.get().strip() == "mahesh" and password.get() == "mahesh":
            login_window_close()
        else:
            messagebox.showerror(
                "Access Denied",
                "Invalid username or password.",
                parent=login
            )
            password.delete(0, tk.END)
            password.focus_set()

    login_button = tk.Button(
        form,
        text="LOGIN TO DASHBOARD",
        command=authenticate_user,
        bg="#2563EB",
        fg="white",
        activebackground="#1D4ED8",
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        font=("Segoe UI", 11, "bold"),
        height=2
    )
    login_button.pack(fill="x", pady=(25, 15))

    tk.Label(
        form,
        text="🔒  Protected administrator access",
        bg="#F7FAFC",
        fg="#94A3B8",
        font=("Segoe UI", 9)
    ).pack()

    username.bind("<Return>", lambda e: authenticate_user())
    password.bind("<Return>", lambda e: authenticate_user())
    username.focus_set()


# ============================================================
# MAIN DASHBOARD
# ============================================================

root.withdraw()

# ---------- Sidebar ----------
sidebar = tk.Frame(root, bg=COLORS["navy"], width=220)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

brand = tk.Frame(sidebar, bg=COLORS["navy"], height=104)
brand.pack(fill="x")
brand.pack_propagate(False)

tk.Label(
    brand,
    text="⬡",
    bg=COLORS["navy"],
    fg="#20A4FF",
    font=("Segoe UI Symbol", 34, "bold")
).pack(side="left", padx=(18, 8), pady=22)

brand_text = tk.Frame(brand, bg=COLORS["navy"])
brand_text.pack(side="left", pady=21)

tk.Label(
    brand_text,
    text="CyberGuard AI",
    bg=COLORS["navy"],
    fg="white",
    font=("Segoe UI", 13, "bold")
).pack(anchor="w")

tk.Label(
    brand_text,
    text="Threat Detection & Analytics",
    bg=COLORS["navy"],
    fg="#78A9D8",
    font=("Segoe UI", 7)
).pack(anchor="w")

sidebar_item(sidebar, "⌂", "Dashboard", True)
sidebar_item(sidebar, "▥", "Analytics", command=lambda: show_all_results())
sidebar_item(sidebar, "⌁", "Model Results", command=lambda: show_all_results())
sidebar_item(sidebar, "◷", "History", command=lambda: print_output("Prediction history is shown in the console."))
sidebar_item(sidebar, "⚙", "Settings", command=lambda: messagebox.showinfo("Settings", "Dashboard settings are ready for customization."))
sidebar_item(sidebar, "●", "About", command=lambda: messagebox.showinfo(
    "About",
    "Cyber Threat Detection\nAI Security Analytics\n\nSVM • ANN • KNN • LSTM • Naive Bayes"
))

# Decorative lower shield
shield_area = tk.Frame(sidebar, bg=COLORS["navy"])
shield_area.pack(side="bottom", fill="both", expand=True)

canvas = tk.Canvas(
    shield_area,
    bg=COLORS["navy"],
    highlightthickness=0
)
canvas.pack(fill="both", expand=True)

canvas.create_oval(25, 100, 195, 270, outline="#0A5AA5", width=2)
canvas.create_oval(43, 118, 177, 252, outline="#0D3E73", width=1)
canvas.create_text(
    110, 175,
    text="⬢",
    fill="#1598F4",
    font=("Segoe UI Symbol", 68)
)
canvas.create_text(
    110, 295,
    text="SECURE\nANALYZE\nSTAY AHEAD",
    fill="#5DA6DF",
    font=("Segoe UI", 8, "bold"),
    justify="center"
)
canvas.create_text(
    22, 350,
    text="v1.0.0",
    fill="#9AB1CC",
    font=("Segoe UI", 8),
    anchor="w"
)


# ---------- Main area ----------
main = tk.Frame(root, bg=COLORS["bg"])
main.pack(side="left", fill="both", expand=True)


# ---------- Header ----------
header = tk.Frame(main, bg=COLORS["navy2"], height=110)
header.pack(fill="x")
header.pack_propagate(False)

# Decorative network background
header_canvas = tk.Canvas(
    header,
    bg=COLORS["navy2"],
    highlightthickness=0
)
header_canvas.place(relx=0.55, rely=0, relwidth=0.45, relheight=1)

for i in range(22):
    x = 30 + (i * 61) % 600
    y = 15 + (i * 37) % 95
    r = 2 if i % 3 else 3
    header_canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill="#2D9BFF",
        outline=""
    )

for i in range(11):
    x1 = 40 + (i * 53) % 570
    y1 = 18 + (i * 19) % 80
    x2 = x1 + 80
    y2 = y1 + 25
    header_canvas.create_line(
        x1, y1, x2, y2,
        fill="#164A83",
        width=1
    )

title_box = tk.Frame(header, bg=COLORS["navy2"])
title_box.pack(side="left", padx=25, pady=14)

tk.Label(
    title_box,
    text="Cyber Threat Detection Based on Artificial Neural Networks",
    bg=COLORS["navy2"],
    fg="white",
    font=("Segoe UI", 19, "bold")
).pack(anchor="w")

tk.Label(
    title_box,
    text="Using Event Profiles",
    bg=COLORS["navy2"],
    fg="#21C8F7",
    font=("Segoe UI", 19, "bold")
).pack(anchor="w")

tk.Label(
    title_box,
    text="Intelligent Event Profile Threat Detection & Analytics",
    bg=COLORS["navy2"],
    fg="#9CB9DA",
    font=("Segoe UI", 9)
).pack(anchor="w", pady=(2, 0))

# Right header info
header_right = tk.Frame(header, bg=COLORS["navy2"])
header_right.pack(side="right", padx=24, pady=14)

online_box = tk.Frame(header_right, bg="#123F63")
online_box.pack(anchor="e")

tk.Label(
    online_box,
    text="● SYSTEM ONLINE",
    bg="#123F63",
    fg="#5DF39D",
    font=("Segoe UI", 9, "bold"),
    padx=12,
    pady=5
).pack()

date_var = tk.StringVar()
time_var = tk.StringVar()

tk.Label(
    header_right,
    textvariable=date_var,
    bg=COLORS["navy2"],
    fg="#D9E8FA",
    font=("Segoe UI", 8)
).pack(anchor="e", pady=(8, 0))

tk.Label(
    header_right,
    textvariable=time_var,
    bg=COLORS["navy2"],
    fg="#8FB0D3",
    font=("Segoe UI", 8)
).pack(anchor="e")


# ---------- Content ----------
content = tk.Frame(main, bg=COLORS["bg"])
content.pack(fill="both", expand=True, padx=17, pady=15)


# ---------- Status cards ----------
cards = tk.Frame(content, bg=COLORS["bg"])
cards.pack(fill="x")

model_status_var = tk.StringVar(value="Ready")
dataset_var = tk.StringVar(value="Not Loaded")
dataset_detail_var = tk.StringVar(value="Upload event log dataset")
filename_var = tk.StringVar(value="")

ai_value_var = tk.StringVar(value="6 Models")
ai_detail_var = tk.StringVar(value="SVM, ANN, KNN, LSTM, NB")

analytics_value_var = tk.StringVar(value="Real-time")
analytics_detail_var = tk.StringVar(value="Visualize threat insights")

model_card = make_card(
    cards,
    "Model Status",
    model_status_var,
    tk.StringVar(value="All systems operational"),
    "⬡",
    COLORS["green"]
)
model_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

dataset_card = make_card(
    cards,
    "Dataset",
    dataset_var,
    dataset_detail_var,
    "▣",
    COLORS["purple"]
)
dataset_card.pack(side="left", fill="both", expand=True, padx=8)

ai_card = make_card(
    cards,
    "AI Models",
    ai_value_var,
    ai_detail_var,
    "♙",
    COLORS["blue"]
)
ai_card.pack(side="left", fill="both", expand=True, padx=8)

analytics_card = make_card(
    cards,
    "Analytics",
    analytics_value_var,
    analytics_detail_var,
    "▥",
    COLORS["orange"]
)
analytics_card.pack(side="left", fill="both", expand=True, padx=(8, 0))


# ---------- Console ----------
console_card = tk.Frame(
    content,
    bg=COLORS["console"],
    highlightbackground="#153E61",
    highlightthickness=1
)
console_card.pack(fill="both", expand=False, pady=(14, 12))
console_card.configure(height=260)
console_card.pack_propagate(False)

console_header = tk.Frame(console_card, bg=COLORS["console2"], height=38)
console_header.pack(fill="x")
console_header.pack_propagate(False)

tk.Label(
    console_header,
    text="✣  Security Analysis Console",
    bg=COLORS["console2"],
    fg="white",
    font=("Segoe UI", 10, "bold")
).pack(side="left", padx=14)

tk.Button(
    console_header,
    text="▣  Clear",
    command=clear_console,
    bg="#173C60",
    fg="white",
    activebackground="#22577F",
    activeforeground="white",
    relief="flat",
    bd=0,
    cursor="hand2",
    font=("Segoe UI", 8, "bold")
).pack(side="right", padx=10, pady=6)

output_text = ScrolledText(
    console_card,
    font=("Consolas", 9),
    bg=COLORS["console"],
    fg=COLORS["console_text"],
    insertbackground="white",
    selectbackground="#145B90",
    relief="flat",
    bd=0,
    padx=18,
    pady=12,
    wrap="word"
)
output_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
output_text.configure(state="disabled")


# ---------- Control center ----------
control_card = tk.Frame(
    content,
    bg="white",
    highlightbackground=COLORS["border"],
    highlightthickness=1
)
control_card.pack(fill="x", pady=(0, 12))

control_top = tk.Frame(control_card, bg="white")
control_top.pack(fill="x", padx=17, pady=(9, 6))

tk.Label(
    control_top,
    text="⚙  Control Center",
    bg="white",
    fg=COLORS["text"],
    font=("Segoe UI", 11, "bold")
).pack(side="left")

tk.Label(
    control_top,
    text="⚡ Run machine learning models and visualize results",
    bg="white",
    fg=COLORS["muted"],
    font=("Segoe UI", 8)
).pack(side="right")

buttons_area = tk.Frame(control_card, bg="white")
buttons_area.pack(fill="x", padx=14, pady=(0, 12))

button_specs = [
    ("Upload Dataset", upload_dataset, COLORS["blue"], "▣"),
    ("Run TF-IDF", tfidf_preprocessing, COLORS["green"], "▤"),
    ("Generate Event Vector", create_event_vectors, COLORS["purple"], "♧"),
    ("Run LSTM", run_svm, COLORS["red"], "⌁"),
    ("Run ANN", run_ann, COLORS["orange"], "♙"),
    ("Run SVM", run_lstm, "#5A50F5", "◷"),
    ("Run KNN", run_knn, "#0B9DB8", "♧"),
    ("Naive Bayes", run_naive_bayes, COLORS["pink"], "∿"),
    ("Accuracy Graph", lambda: draw_graph("Accuracy"), "#287EEA", "▥"),
    ("Precision / Recall / F1", lambda: draw_graph("Precision"), COLORS["green"], "◔"),
    ("Show Results", show_all_results, COLORS["purple"], "▤"),
]

for index, (text, command, bg, icon) in enumerate(button_specs):
    btn = make_button(buttons_area, text, command, bg, icon)
    row = index // 6
    col = index % 6
    btn.grid(row=row, column=col, padx=4, pady=4, sticky="ew")
    buttons_area.grid_columnconfigure(col, weight=1)


# ---------- Threat input ----------
input_card = tk.Frame(
    content,
    bg="white",
    highlightbackground=COLORS["border"],
    highlightthickness=1
)
input_card.pack(fill="x")

input_top = tk.Frame(input_card, bg="white")
input_top.pack(fill="x", padx=17, pady=(8, 2))

tk.Label(
    input_top,
    text="⬡  Threat Event Input",
    bg="white",
    fg=COLORS["text"],
    font=("Segoe UI", 10, "bold")
).pack(side="left")

tk.Label(
    input_top,
    text="ⓘ  Enter a log/event message to predict the threat category",
    bg="white",
    fg=COLORS["muted"],
    font=("Segoe UI", 8)
).pack(side="right")

input_row = tk.Frame(input_card, bg="white")
input_row.pack(fill="x", padx=16, pady=(2, 9))

input_entry = tk.Entry(
    input_row,
    font=("Segoe UI", 10),
    bg="#F8FAFD",
    fg=COLORS["text"],
    insertbackground=COLORS["text"],
    relief="solid",
    bd=1
)
input_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))

# Placeholder
placeholder = "e.g. Failed login attempt from 192.168.1.10, multiple authentication failures..."
input_entry.insert(0, placeholder)
input_entry.configure(fg="#91A2BB")


def focus_in(event):
    if input_entry.get() == placeholder:
        input_entry.delete(0, tk.END)
        input_entry.configure(fg=COLORS["text"])


def focus_out(event):
    if not input_entry.get().strip():
        input_entry.insert(0, placeholder)
        input_entry.configure(fg="#91A2BB")


def prediction_click():
    if input_entry.get() == placeholder:
        print_output("Please enter an actual threat event.")
        return
    predict_event()


input_entry.bind("<FocusIn>", focus_in)
input_entry.bind("<FocusOut>", focus_out)
input_entry.bind("<Return>", lambda e: prediction_click())

predict_button = tk.Button(
    input_row,
    text="⌕  Predict Threat",
    command=prediction_click,
    bg="#1688FF",
    fg="white",
    activebackground="#0875E0",
    activeforeground="white",
    relief="flat",
    bd=0,
    cursor="hand2",
    font=("Segoe UI", 10, "bold"),
    padx=22,
    pady=9
)
predict_button.pack(side="right")


# ---------- Footer ----------
footer = tk.Frame(main, bg=COLORS["bg"], height=22)
footer.pack(fill="x")
footer.pack_propagate(False)

tk.Label(
    footer,
    text="Cyber Threat Detection  |  AI Security Analytics  |  Stay Safe  |  Build a Secure Tomorrow",
    bg=COLORS["bg"],
    fg="#7184A5",
    font=("Segoe UI", 7)
).pack(side="right", padx=18)


# ============================================================
# START
# ============================================================

update_clock()
show_login()
root.mainloop()
