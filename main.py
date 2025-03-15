import os
import base64
import io
import uvicorn

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

# For the Inception-based model
from torchvision import models
import torchvision.transforms as T

# For the MLP-based Dst model
import pandas as pd
import requests

# ===========================
# 0) ENV & CONFIG
# ===========================
load_dotenv()

PORT = int(os.getenv("PORT", 9092))
HOST = os.getenv("HOST", "0.0.0.0")
inception_model_path = os.getenv("MODEL_PATH", "models/best_inception_9classes.pt")

# We'll check the existence of the Inception model file:
if not os.path.isfile(inception_model_path):
    raise FileNotFoundError(f"Model file '{inception_model_path}' not found at '{os.getcwd()}'")

# The MLP Dst model & scaler are assumed to be named:
DST_MODEL_FILE = "models/dst_model.pt"
SCALER_FILE = "models/scaler.joblib"

if not os.path.isfile(DST_MODEL_FILE):
    raise FileNotFoundError(f"Dst model file '{DST_MODEL_FILE}' not found!")
if not os.path.isfile(SCALER_FILE):
    raise FileNotFoundError(f"Scaler file '{SCALER_FILE}' not found!")


# ===========================
# 1) SKIN LESION MODEL SETUP
# ===========================
classes_9 = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

inference_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

def load_inception_model(model_path):
    """
    Load an Inception v3 model with custom heads for 9 classes.
    """
    model = models.inception_v3(weights=None, aux_logits=True, init_weights=True)
    # Modify AuxLogits
    model.AuxLogits.fc = nn.Linear(768, 9)
    # Modify final fc
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 9)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ===========================
# 2) Dst MLP MODEL SETUP
# ===========================
class LSTMModel(nn.Module):
    def __init__(self, num_features=14, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        batch_size = x.size(0)
        out, (h, c) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        # take last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# We'll also define a Pydantic model for solar wind data
class SolarWindRequest(BaseModel):
    bx_gse: float
    by_gse: float
    bz_gse: float
    theta_gse: float
    phi_gse: float
    bx_gsm: float
    by_gsm: float
    bz_gsm: float
    theta_gsm: float
    phi_gsm: float
    bt: float
    density: float
    speed: float
    temperature: float


FEATURE_COLS = [
    "bx_gse","by_gse","bz_gse","theta_gse","phi_gse",
    "bx_gsm","by_gsm","bz_gsm","theta_gsm","phi_gsm",
    "bt","density","speed","temperature"
]


# ===========================
# 3) FASTAPI APP
# ===========================
app = FastAPI()

# We'll store the loaded models & scaler in global variables
uv_model = None       # Inception
dst_model = None      # MLP for Dst
scaler = None         # joblib scaler for Dst features

# ===========================
# 4) STARTUP EVENT
# ===========================
@app.on_event("startup")
def on_startup():
    global uv_model, dst_model, scaler

    # 4A) Load Inception for skin lesions
    uv_model = load_inception_model(inception_model_path)
    uv_model.eval()
    print(f"Inception model loaded from {inception_model_path}")

    # 4B) Load MLP for Dst
    dst_model_obj = LSTMModel(num_features=14, hidden_size=64)
    dst_model_obj.load_state_dict(torch.load(DST_MODEL_FILE, map_location=torch.device("cpu")))
    dst_model_obj.eval()
    dst_model = dst_model_obj
    print(f"Dst model loaded from {DST_MODEL_FILE}")

    # 4C) Load scaler
    scaler_obj = joblib.load(SCALER_FILE)
    scaler = scaler_obj
    print(f"Scaler loaded from {SCALER_FILE}")

    print("All models & scaler initialized.")


# ===========================
# 5) IMAGE DIAGNOSIS ENDPOINT
# ===========================
@app.post("/ml/diagnosis")
async def predict_skin_lesion(request: dict):
    """
    Expects a JSON body with key 'image' containing a base64-encoded image.
    Returns the predicted 9-class label + probabilities.
    """
    try:
        image_data = request.get("image")
        if not image_data:
            return {"error": "No image data provided"}

        # If data URL format, split off the "data:image/xxx;base64,"
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor = inference_transform(image).unsqueeze(0)  # shape (1,3,299,299)

        with torch.no_grad():
            output = uv_model(input_tensor)
            probs = torch.softmax(output, dim=1)

        _, predicted_idx = torch.max(output, 1)
        predicted_label = classes_9[predicted_idx.item()]

        class_probabilities = {
            f"{cls}": f"{prob.item() * 100:.2f}%" 
            for cls, prob in zip(classes_9, probs[0])
        }
        class_probabilities["predicted_class"] = predicted_label

        return class_probabilities

    except Exception as e:
        return {"error": str(e)}

@app.get("/ml/dst")
def predict_dst_ace():
    """
    Automatically fetches ACE data from NOAA SWPC,
    merges magnetometer & swepam, picks the latest row,
    constructs a single sequence for the LSTM,
    and returns the predicted Dst (no user parameters).
    """
    # 1) Download magnetometer data (1-minute cadence)
    url_mag = "https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1h.json"
    r_mag = requests.get(url_mag)
    data_mag = r_mag.json()
    df_mag = pd.DataFrame(data_mag)

    # 2) Download swepam data (1-minute cadence)
    url_swepam = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json"
    r_swepam = requests.get(url_swepam)
    data_swepam = r_swepam.json()
    df_swepam = pd.DataFrame(data_swepam)

    if df_mag.empty or df_swepam.empty:
        return {"error": "No data returned from ACE endpoints."}

    # 3) Convert 'time_tag' to datetime, sort, then merge on 'time_tag'
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
    df_swepam['time_tag'] = pd.to_datetime(df_swepam['time_tag'])

    df_merged = pd.merge(df_mag, df_swepam, on='time_tag', how='inner')
    if df_merged.empty:
        return {"error": "No overlapping mag/swepam data to merge."}

    df_merged.sort_values('time_tag', inplace=True)

    # 4) Grab the latest row (this is your "current" solar wind reading)
    last_row = df_merged.iloc[-1]

    # 5) Map the NOAA columns to your 14 feature columns
    #    This is an example mapping; adapt to real NOAA fields
    #    The placeholders (0.0) are used if NOAA doesn't provide them
    df_input = pd.DataFrame([{
        # If you want bx_gse but NOAA only has bx_gsm, you must decide how to handle that.
        # For now, let's assume you only have gsm in NOAA, so we do placeholder for gse:
        "bx_gse": 0.0,
        "by_gse": 0.0,
        "bz_gse": 0.0,
        "theta_gse": 0.0,
        "phi_gse": 0.0,

        # NOAA might have these as 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt'
        "bx_gsm":  last_row.get("bx_gsm", 0.0),
        "by_gsm":  last_row.get("by_gsm", 0.0),
        "bz_gsm":  last_row.get("bz_gsm", 0.0),
        "theta_gsm": 0.0,   # placeholder
        "phi_gsm":   0.0,   # placeholder

        "bt": last_row.get("bt", 0.0),

        # swepam fields might be named 'proton_density', 'bulk_speed', 'ion_temperature'
        "density": last_row.get("proton_density", 0.0),
        "speed":   last_row.get("bulk_speed", 0.0),
        "temperature": last_row.get("ion_temperature", 0.0)
    }])

    # 6) Scale it using the same 'scaler' loaded on startup
    X_scaled = scaler.transform(df_input[FEATURE_COLS])

    # 7) Shape it for LSTM: (batch=1, seq_len=1, num_features=14)
    X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0)

    # 8) Run inference
    with torch.no_grad():
        pred = dst_model(X_tensor).squeeze().item()

    return {
        "time_tag": str(last_row['time_tag']),
        "predicted_dst": pred
    }

# ===========================
# 7) RUN THE APP (OPTIONAL)
# ===========================
# If you want to run with uvicorn programmatically:


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)