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
    import numpy as np
    from PIL import ImageFilter

    image = image.convert("L")

    # 1. Find the bounding box of the drawn digit (non-white pixels)
    img_array = np.array(image)
    # Threshold: treat pixels < 250 as "ink"
    coords = np.column_stack(np.where(img_array < 250))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Crop to the digit with a small padding
        pad = 15
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(image.height, y_max + pad)
        x_max = min(image.width, x_max + pad)
        image = image.crop((x_min, y_min, x_max, y_max))

    # 2. Invert: drawn image has black-on-white, MNIST expects white-on-black
    image = ImageOps.invert(image)

    # 3. Apply slight blur to smooth jagged edges from drawing
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    # 4. Resize to 20x20 (standard MNIST digit size), then paste onto 28x28 center
    image = image.resize((20, 20), Image.Resampling.LANCZOS)
    new_image = Image.new("L", (28, 28), 0)
    new_image.paste(image, (4, 4))
    image = new_image

    # 5. Apply MNIST normalization
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