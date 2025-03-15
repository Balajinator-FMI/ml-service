import base64
import torch
from fastapi import FastAPI
from dotenv import load_dotenv
import os
import torch.nn as nn
from torchvision import models
import io
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

load_dotenv()

PORT = int(os.getenv("PORT", 9092))
HOST = os.getenv("HOST", "0.0.0.0")
model_path = os.getenv("MODEL_PATH", "models/best_inception_9classes.pt")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found in the current directory!")

def load_model():
    model = models.inception_v3(weights=None, aux_logits=True)
    model.AuxLogits.fc = nn.Linear(768, 9)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 9)
    )
    model.load_state_dict(torch.load("models/best_inception_9classes.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

app = FastAPI()

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

@app.post('/ml/diagnosis')
async def predict(request: dict):
    try:
        image_data = request.get("image")

        if not image_data:
            return {"error": "No image data provided"}

        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor = inference_transform(image).unsqueeze(0).to(torch.device('cpu'))

        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)

        _, predicted_idx = torch.max(output, 1)
        predicted_label = classes_9[predicted_idx.item()]

        class_probabilities = {
            f"{i}": f"{prob * 100:.2f}%" for i, prob in zip(classes_9, probs[0])
        }

        class_probabilities["predicted_class"] = predicted_label

        return class_probabilities

    except Exception as e:
        return {"error": str(e)}
