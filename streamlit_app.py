
import streamlit as st
from ultralytics import YOLO
from pathlib import Path

import torch
import torch.nn as nn
import json
import gc

from PIL import Image
from torchvision import transforms, models


# -------------------------------------------------
# 1. Streamlit configuration
# -------------------------------------------------

st.set_page_config(
    page_title="PulseVue - Chickpea Detection",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "pulsevue_augmented" /
    "weights" / "best.pt"
)

CLASSIFIER_PATH = BASE_DIR / "chickpea_classifier.pth"
CLASS_INFO_PATH = BASE_DIR / "classifier_classes.json"

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

CHICKPEA_CLASSES = {0, 1, 2, 3, 4, 7, 8}
FOREIGN_OBJECT_CLASSES = {5, 6}


# -------------------------------------------------
# 2. Load trained models
# -------------------------------------------------

@st.cache_resource
def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_PATH}"
        )

    if not CLASS_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Class information not found: {CLASS_INFO_PATH}"
        )

    yolo_model = YOLO(str(MODEL_PATH))

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

    return yolo_model, classifier, class_names


# -------------------------------------------------
# 3. Image preprocessing
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
# 4. Validate chickpea image
# -------------------------------------------------

def validate_chickpea_image(image, classifier, class_names):

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():

        outputs = classifier(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    predicted_label = class_names[predicted_class.item()]

    return predicted_label, confidence.item()


# -------------------------------------------------
# 5. Application interface
# -------------------------------------------------

st.title("🌱 PulseVue")
st.subheader("Smart Chickpea Detection and Quality Analysis")

st.write(
    "Upload a chickpea image to detect and count chickpeas "
    "and identify foreign materials using deep learning."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a chickpea image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG. Maximum size: 10 MB."
)


# -------------------------------------------------
# 6. Image analysis
# -------------------------------------------------

if uploaded_file is not None:

    if uploaded_file.size > 10 * 1024 * 1024:

        st.error("Image size must be 10 MB or less!")

    else:

        try:

            original_image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original_image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            with col2:
                st.info("Your image is ready for analysis.")

            if st.button("Analyse Image", type="primary"):

                with st.spinner("Loading models and analysing image..."):

                    yolo_model, classifier, class_names = load_models()

                    # STEP 1: MobileNetV2 validation

                    predicted_label, confidence = (
                        validate_chickpea_image(
                            original_image,
                            classifier,
                            class_names
                        )
                    )

                    # STEP 2: Reject non-chickpea images

                    if predicted_label != "chickpea":

                        st.error(
                            "Invalid Image: Please upload a chickpea image."
                        )

                        st.write(
                            f"Classification confidence: "
                            f"{confidence * 100:.2f}%"
                        )

                    else:

                        # STEP 3: YOLOv8 detection

                        predictions = yolo_model.predict(
                            source=original_image,
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

                        del predictions

                        # STEP 4: Get detected class IDs

                        detected_class_ids = []

                        if result.boxes is not None:

                            detected_class_ids = (
                                result.boxes.cls.int().cpu().tolist()
                            )

                        # STEP 5: Count chickpeas

                        chickpea_count = sum(
                            1 for class_id in detected_class_ids
                            if class_id in CHICKPEA_CLASSES
                        )

                        # STEP 6: Check foreign materials

                        foreign_object_detected = any(
                            class_id in FOREIGN_OBJECT_CLASSES
                            for class_id in detected_class_ids
                        )

                        # STEP 7: Generate annotated image

                        annotated_image = result.plot()

                        annotated_image = annotated_image[:, :, ::-1].copy()

                        del result
                        gc.collect()

                        # STEP 8: Display results

                        st.divider()

                        st.header("Analysis Results")

                        result_col1, result_col2 = st.columns(2)

                        with result_col1:

                            st.image(
                                original_image,
                                caption="Original Image",
                                use_container_width=True
                            )

                        with result_col2:

                            st.image(
                                annotated_image,
                                caption="Detected Chickpeas",
                                use_container_width=True
                            )

                        st.divider()

                        metric1, metric2 = st.columns(2)

                        with metric1:
                            st.metric(
                                "Total Chickpeas Detected",
                                chickpea_count
                            )

                        with metric2:

                            if foreign_object_detected:
                                st.error("Foreign Materials Detected")
                            else:
                                st.success("No Foreign Materials Detected")

                        st.success("Image analysis completed successfully!")

                        del annotated_image
                        gc.collect()

        except Exception as error:

            st.error(
                "Unable to complete image analysis. "
                "Please check the image and try again."
            )

            st.exception(error)

            gc.collect()


st.divider()

st.caption("PulseVue | AI-Based Chickpea Detection and Quality Analysis")
