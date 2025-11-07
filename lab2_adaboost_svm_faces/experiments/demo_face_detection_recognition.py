#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.orl_faces import load_orl_faces
from src.detection.haar_detector import HaarFaceDetector, FaceBoundingBox
from src.pipeline.recognizer import PCASVMRecognizer, MODEL_FILE

MODELS_DIR = ROOT / "models"
ASSETS_FIG = ROOT / "assets" / "figures"
ASSETS_SAMPLES = ROOT / "assets" / "samples"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_SAMPLES.mkdir(parents=True, exist_ok=True)


def _ensure_model(model_path: Path) -> PCASVMRecognizer:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file {model_path} not found. Run train_pca_svm_recognizer.py first."
        )
    return PCASVMRecognizer.load(model_path)


def _generate_sample_image(index: int = 0) -> Path:
    sample_path = ASSETS_SAMPLES / "orl_sample_face.png"
    if sample_path.exists():
        return sample_path

    faces = load_orl_faces(shuffle=True, random_state=0)
    image = (faces.images[index] * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    enlarged = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(sample_path), enlarged)
    return sample_path


def _prepare_face_patch(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    face_roi = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, -1)


def run_demo(image_path: Optional[Path], model_path: Path) -> Path:
    if image_path is None:
        image_path = _generate_sample_image()

    recognizer = _ensure_model(model_path)
    detector = HaarFaceDetector(min_size=(40, 40))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    detections = detector.detect(image)
    if not detections:
        h, w = image.shape[:2]
        fallback_box = (int(0.15 * w), int(0.15 * h), int(0.7 * w), int(0.7 * h))
        detections = [FaceBoundingBox(*fallback_box)]

    face_patches = [_prepare_face_patch(image, det.to_tuple()) for det in detections]
    X = np.vstack(face_patches)
    predictions = recognizer.predict(X)
    for det, pred in zip(detections, predictions):
        x, y, w, h = det.to_tuple()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{pred.label} ({pred.confidence * 100:.1f}%)"
        cv2.putText(image, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = ASSETS_FIG / "face_detection_demo.png"
    cv2.imwrite(str(output_path), image)
    print("Annotated image saved to:", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face detection + recognition demo")
    parser.add_argument("image", nargs="?", type=Path, help="Path to input image")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / MODEL_FILE, help="Path to trained recognizer model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args.image, args.model)


if __name__ == "__main__":
    main()
