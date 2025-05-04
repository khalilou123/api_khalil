from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
import uvicorn
import os

app = FastAPI()

# Ajoute le middleware CORS en premier
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],  # Autorise les deux origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
try:
    model = tf.keras.models.load_model("belle_voix_model.h5")
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
FIXED_TIME_STEPS = 130

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        else:
            y = y[:SAMPLES_PER_TRACK]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        if mel_db.shape[1] < FIXED_TIME_STEPS:
            mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_TIME_STEPS - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :FIXED_TIME_STEPS]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db[np.newaxis, ..., np.newaxis]
    except Exception as e:
        raise ValueError(f"Erreur de prétraitement : {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("Requête reçue pour /predict")
    try:
        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        features = preprocess_audio(temp_path)
        print(f"Forme des features : {features.shape}")
        prediction = model.predict(features)[0][0]
        label = "belle_voix" if prediction > 0.5 else "non_belle_voix"
        os.remove(temp_path)
        return {"prediction": label, "confidence": float(prediction)}
    except Exception as e:
        print(f"Erreur : {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))