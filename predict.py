from pathlib import Path
from typing import Union

import torch
from PIL import Image, ImageOps

from model import CNN

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "mnist.pth"
_MODEL = None
_MODEL_DEVICE = None


def get_model(weights_path: Union[str, Path] = MODEL_PATH):
    global _MODEL, _MODEL_DEVICE
    if _MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN().to(device)
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        _MODEL_DEVICE = device
    return _MODEL, _MODEL_DEVICE


def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    pixels = list(image.getdata())
    mean_value = sum(pixels) / len(pixels) if pixels else 255.0
    if mean_value < 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = image.convert("L")

    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 1, 28, 28) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    return tensor


def predict_from_pil(image: Image.Image):
    model, device = get_model()
    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted].item())
    return predicted, confidence


def predict_from_path(image_path: Union[str, Path]):
    image = Image.open(image_path)
    return predict_from_pil(image)


if __name__ == "__main__":
    sample_path = Path("data") / "sample.png"
    if sample_path.exists():
        prediction, confidence = predict_from_path(sample_path)
        print(f"Predicted digit: {prediction} (confidence: {confidence:.2%})")
    else:
        print("No sample image found. Please provide an image file.")