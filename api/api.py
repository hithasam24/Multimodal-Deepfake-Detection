from fastapi import FastAPI, UploadFile, File
import tempfile
import torch
import numpy as np
import cv2
import librosa
from models.cmgan import CMGANOnlyModel

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CMGANOnlyModel().to(DEVICE)
# Example:
# model.load_state_dict(torch.load("outputs/checkpoints/best_cmgan_full.pt", map_location=DEVICE))
model.eval()


def preprocess(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(6):
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((160, 160, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (160, 160))
        frames.append(frame)
    cap.release()

    frames = np.array(frames).astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))
    frames = torch.tensor(frames).unsqueeze(0).float().to(DEVICE)

    y, _ = librosa.load(video_path, sr=16000, mono=True, duration=3.0)
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return frames, mel


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    frames, mel = preprocess(tmp_path)
    with torch.no_grad():
        outputs = model(frames, mel)
        prob = torch.sigmoid(outputs["logits"]).item()

    return {
        "prediction": "Fake" if prob > 0.5 else "Real",
        "confidence": float(prob),
    }
