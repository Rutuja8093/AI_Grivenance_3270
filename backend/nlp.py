import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# ----- AI Libraries -----
try:
    import easyocr
    reader = easyocr.Reader(['en'])
except ImportError:
    print("⚠️ EasyOCR not installed, image complaints will fail")
    reader = None

try:
    import whisper
    model_whisper = whisper.load_model("base")
except ImportError:
    print("⚠️ Whisper not installed, audio complaints will fail")
    model_whisper = None

# ----- Paths -----
DATA_PATH = "data/complaints.csv"
MODEL_PATH = "models/grievance_model.pkl"

# ----- Train AI model -----
def train_model():
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ {DATA_PATH} not found. Add dataset first.")
        return

    df = pd.read_csv(DATA_PATH)

    required_cols = ["Complaint_Text", "Category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    X = df["Complaint_Text"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

# ----- Predict category -----
def predict_category(text):
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}, returning 'Unknown'")
        return "Unknown"
    model = joblib.load(MODEL_PATH)
    return model.predict([text])[0]

# ----- Extract text from image -----
def extract_text_from_image(image_path):
    if reader is None:
        raise RuntimeError("EasyOCR not available")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

# ----- Transcribe audio -----
def transcribe_audio(audio_path):
    if model_whisper is None:
        raise RuntimeError("Whisper not available")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    result = model_whisper.transcribe(audio_path)
    return result.get("text", "")
