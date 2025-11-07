#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.orl_faces import load_orl_faces
from src.pipeline.recognizer import PCASVMRecognizer, MODEL_FILE

MODELS_DIR = ROOT / "models"
ASSETS_TAB = ROOT / "assets" / "tables"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 0) -> None:
    faces = load_orl_faces(shuffle=True, random_state=random_state)
    recognizer = PCASVMRecognizer(pca_components=120, C=10.0, gamma="scale", random_state=random_state)
    recognizer.fit(faces.data, faces.target)

    model_path = MODELS_DIR / MODEL_FILE
    recognizer.save(model_path)

    metadata = {
        "train_samples": int(faces.data.shape[0]),
        "n_features": int(faces.data.shape[1]),
        "n_classes": int(np.unique(faces.target).size),
        "pca_components": 120,
        "svm_C": 10.0,
        "svm_gamma": "scale",
        "model_path": str(model_path.name),
    }
    (ASSETS_TAB / "recognizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model saved to:", model_path)
    print("Metadata recorded at:", ASSETS_TAB / "recognizer_metadata.json")


if __name__ == "__main__":
    main()
