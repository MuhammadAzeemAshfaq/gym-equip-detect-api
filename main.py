from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="Gym Equipment Detection API",
    description="YOLOv8-based gym equipment detection for FYP",
    version="1.0"
)

# Load YOLO model once when server starts
model = YOLO("best.pt")

# Store class ID → name mapping
CLASS_NAMES = model.names

# Print mapping once for confirmation (optional but useful)
print("Loaded class mapping:", CLASS_NAMES)


@app.get("/")
def home():
    return {"message": "Gym Equipment Detection API Running"}


@app.get("/classes")
def get_classes():
    """
    Returns class ID to equipment name mapping
    Useful for debugging / documentation
    """
    return CLASS_NAMES


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts an image and returns the detected gym equipment
    """

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLO inference
    results = model(image)
    boxes = results[0].boxes

    # No detection case
    if boxes is None or len(boxes) == 0:
        return {
            "detected": False,
            "equipment": None,
            "confidence": None
        }

    # Pick the detection with highest confidence
    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])

    return {
        "detected": True,
        "class_id": class_id,
        "equipment": CLASS_NAMES[class_id],
        "confidence": confidence
    }
