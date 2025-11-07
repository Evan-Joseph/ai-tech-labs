from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


def _default_cascade_path() -> Path:
    try:
        return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    except AttributeError:
        root = Path(cv2.__file__).resolve().parent
        candidate = root / "data" / "haarcascade_frontalface_default.xml"
        if candidate.exists():
            return candidate
        alt = root.parent / "share" / "opencv4" / "haarcascades" / "haarcascade_frontalface_default.xml"
        if alt.exists():
            return alt

    storage = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"
    if not storage.exists():
        storage.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(CASCADE_URL, storage)
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                "Failed to locate or download haarcascade_frontalface_default.xml"
            ) from exc
    return storage


DEFAULT_CASCADE_PATH = _default_cascade_path()


@dataclass
class FaceBoundingBox:
    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class HaarFaceDetector:
    def __init__(
        self,
        cascade_path: str | Path = DEFAULT_CASCADE_PATH,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
    ) -> None:
        cascade_path = Path(cascade_path)
        if not cascade_path.exists():
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise FileNotFoundError(f"Failed to load Haar cascade at {cascade_path}")
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(min_size)

    def detect(self, image: np.ndarray) -> List[FaceBoundingBox]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [FaceBoundingBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
