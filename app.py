
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from pathlib import Path

import cv2
import uuid
import torch
import torch.nn as nn
import json
import gc

from PIL import Image
from torchvision import transforms, models


# -------------------------------------------------
# 1. Flask configuration
# -------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Reduce CPU thread usage
torch.set_num_threads(1)


# -------------------------------------------------
# 2. Load YOLO model
# -------------------------------------------------

MODEL_PATH = (
    BASE_DIR / "runs" / "pulsevue_augmented" /
    "weights" / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

yolo_model = YOLO(str(MODEL_PATH))

print("PulseVue YOLO model loaded successfully!")


# -------------------------------------------------
# 3. YOLO class information
# -------------------------------------------------

# Chickpea-related classes:
# 0 = Calibre 7
# 1 = Danado
# 2 = Entretejido
# 3 = Garbanzo
# 4 = Manchado
# 7 = Roto
# 8 = Saludable

CHICKPEA_CLASSES = {0, 1, 2, 3, 4, 7, 8}

# Foreign material classes:
# 5 = Material extrano inorganico
# 6 = Material extrano organico

FOREIGN_OBJECT_CLASSES = {5, 6}


# -------------------------------------------------
# 4. Load MobileNetV2 classifier
# -------------------------------------------------

CLASSIFIER_PATH = BASE_DIR / "chickpea_classifier.pth"
CLASS_INFO_PATH = BASE_DIR / "classifier_classes.json"

DEVICE = torch.device("cpu")

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(
        f"Classifier model not found: {CLASSIFIER_PATH}"
    )

if not CLASS_INFO_PATH.exists():
    raise FileNotFoundError(
        f"Classifier class information not found: {CLASS_INFO_PATH}"
    )

with open(CLASS_INFO_PATH, "r") as file:
    class_mapping = json.load(file)

class_names = [
    name for name, index in
    sorted(class_mapping.items(), key=lambda item: item[1])
]

classifier = models.mobilenet_v2(weights=None)

input_features = classifier.classifier[1].in_features

classifier.classifier[1] = nn.Linear(
    input_features,
    len(class_names)
)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE
    )
)

classifier = classifier.to(DEVICE)
classifier.eval()

print("PulseVue MobileNetV2 classifier loaded successfully!")


# -------------------------------------------------
# 5. Image preprocessing
# -------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# 6. Validate chickpea image
# -------------------------------------------------

def validate_chickpea_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image.thumbnail((1280, 1280))

        image_tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():

        outputs = classifier(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    predicted_label = class_names[predicted_class.item()]

    confidence_value = confidence.item()

    return predicted_label, confidence_value


# -------------------------------------------------
# 7. Allowed image formats
# -------------------------------------------------

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def allowed_file(filename):

    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------
# 8. Home page
# -------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------------------
# 9. Upload page
# -------------------------------------------------

@app.route("/upload")
def upload_page():

    return render_template("upload.html")


# -------------------------------------------------
# 10. Analyse uploaded image
# -------------------------------------------------

@app.route("/analyse", methods=["POST"])
def analyse():

    if "image" not in request.files:
        return "No image uploaded!", 400

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image!", 400

    if not allowed_file(file.filename):
        return "Please upload a JPG, JPEG or PNG image!", 400

    unique_id = uuid.uuid4().hex

    extension = Path(file.filename).suffix.lower()

    uploaded_filename = f"{unique_id}{extension}"

    result_filename = f"{unique_id}.jpg"

    upload_path = UPLOAD_FOLDER / uploaded_filename

    result_path = RESULT_FOLDER / result_filename

    file.save(str(upload_path))


    # ---------------------------------------------
    # STEP 1: Validate image using MobileNetV2
    # ---------------------------------------------

    try:

        predicted_label, confidence = validate_chickpea_image(
            str(upload_path)
        )

    except Exception as error:

        print(f"Classifier error: {error}")

        upload_path.unlink(missing_ok=True)

        gc.collect()

        return (
            "Unable to process this image. Please upload a valid image.",
            400
        )


    # ---------------------------------------------
    # STEP 2: Reject non-chickpea images
    # ---------------------------------------------

    if predicted_label != "chickpea":

        return render_template(
            "invalid_image.html",
            confidence=round(confidence * 100, 2)
        )


    # ---------------------------------------------
    # STEP 3: Analyse valid image using YOLOv8
    # ---------------------------------------------

    try:

        predictions = yolo_model.predict(
            source=str(upload_path),
            conf=0.25,
            imgsz=416,
            max_det=100,
            agnostic_nms=True,
            device="cpu",
            stream=True,
            save=False,
            verbose=False
        )

        result = next(iter(predictions))

        # Release generator reference
        del predictions


        # -----------------------------------------
        # STEP 4: Get detected class IDs
        # -----------------------------------------

        detected_class_ids = []

        if result.boxes is not None:

            detected_class_ids = (
                result.boxes.cls.int().cpu().tolist()
            )


        # -----------------------------------------
        # STEP 5: Count chickpeas
        # -----------------------------------------

        chickpea_count = sum(
            1 for class_id in detected_class_ids
            if class_id in CHICKPEA_CLASSES
        )


        # -----------------------------------------
        # STEP 6: Check foreign material presence
        # -----------------------------------------

        foreign_object_detected = any(
            class_id in FOREIGN_OBJECT_CLASSES
            for class_id in detected_class_ids
        )


        # -----------------------------------------
        # STEP 7: Generate annotated image
        # -----------------------------------------

        annotated_image = result.plot()

        image_saved = cv2.imwrite(
            str(result_path),
            annotated_image
        )

        if not image_saved:

            return "Unable to save the analysis result image.", 500

        del result
        del annotated_image

        gc.collect()


        # -----------------------------------------
        # STEP 8: Display analysis results
        # -----------------------------------------

        return render_template(
            "result.html",

            original_image=url_for(
                "static",
                filename=f"uploads/{uploaded_filename}"
            ),

            result_image=url_for(
                "static",
                filename=f"results/{result_filename}"
            ),

            total_count=chickpea_count,

            foreign_object_detected=foreign_object_detected
        )

    except Exception as error:

        print(f"YOLO analysis error: {error}")

        gc.collect()

        return (
            "Unable to complete image analysis. Please try again.",
            500
        )


# -------------------------------------------------
# 11. File size error
# -------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return "Image size must be 10 MB or less!", 413


# -------------------------------------------------
# 12. Run application
# -------------------------------------------------

if __name__ == "__main__":

    app.run(debug=False)
