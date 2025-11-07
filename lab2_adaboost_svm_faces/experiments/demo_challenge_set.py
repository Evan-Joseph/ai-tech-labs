#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.haar_detector import HaarFaceDetector, FaceBoundingBox
from src.pipeline.recognizer import PCASVMRecognizer, MODEL_FILE

SOURCE_DIR = ROOT / "assets" / "funny_pics"
RESULTS_DIR = ROOT / "assets" / "figures" / "challenge_results"
MODELS_DIR = ROOT / "models"

# Colors (RGB for PIL, BGR for OpenCV)
# Choose GOLD for a professional look
GOLD_RGB = (255, 215, 0)
GOLD_BGR = (0, 215, 255)
WHITE_RGB = (255, 255, 255)
CRIMSON_RGB = (220, 20, 60)

FONT_CANDIDATES_CJK: tuple[str, ...] = (
    "/System/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Songti SC.ttc",
    "/Library/Fonts/Songti.ttc",
    "/Library/Fonts/Songti SC.ttc",
    str(Path.home() / "Library/Fonts/Songti.ttc"),
    str(Path.home() / "Library/Fonts/Songti SC.ttc"),
)

FONT_CANDIDATES_LATIN: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Times.ttc",
    "/Library/Fonts/Times New Roman.ttf",
    str(Path.home() / "Library/Fonts/Times New Roman.ttf"),
)


def _prepare_face_patch(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    face_roi = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, -1)

def _load_font_from_candidates(candidates: tuple[str, ...], size: int, description: str) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    tried = ", ".join(str(Path(c).expanduser()) for c in candidates)
    raise FileNotFoundError(f"无法加载{description}字体，请确认已安装：{tried}")


@lru_cache(maxsize=16)
def _get_fonts(font_size: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    cjk_font = _load_font_from_candidates(FONT_CANDIDATES_CJK, font_size, "苹果宋体 (Songti SC)")
    latin_font = _load_font_from_candidates(FONT_CANDIDATES_LATIN, font_size, "新罗马 (Times New Roman)")
    return cjk_font, latin_font


def _segment_text(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    if not text:
        return segments
    current_type = "latin" if text[0].isascii() else "cjk"
    buffer = [text[0]]
    for ch in text[1:]:
        seg_type = "latin" if ch.isascii() else "cjk"
        if seg_type == current_type:
            buffer.append(ch)
        else:
            segments.append((current_type, "".join(buffer)))
            buffer = [ch]
            current_type = seg_type
    segments.append((current_type, "".join(buffer)))
    return segments


def _render_text_annotations(
    image: np.ndarray,
    annotations: list[tuple[int, int, str, tuple[int, int, int], tuple[int, int, int], float]],
    font_size: int,
) -> np.ndarray:
    if not annotations:
        return image

    cjk_font, latin_font = _get_fonts(font_size)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image).convert("RGBA")
    drawer = ImageDraw.Draw(pil_image)

    # Separate overlay for semi-transparent rectangles
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Determine availability
    has_textbbox = hasattr(drawer, "textbbox")

    pad = max(4, int(0.2 * font_size))

    for x, y, text, text_color, bg_color, alpha in annotations:
        # Measure total text width by segments
        total_width = 0
        max_height = 0
        segments = _segment_text(text)
        for seg_type, seg_text in segments:
            font = latin_font if seg_type == "latin" else cjk_font
            if has_textbbox:
                l, t, r, b = drawer.textbbox((0, 0), seg_text, font=font)
                seg_w, seg_h = r - l, b - t
            else:
                seg_w, seg_h = drawer.textsize(seg_text, font=font)
            total_width += seg_w
            max_height = max(max_height, seg_h)

        # Background rectangle (semi-transparent)
        rect = (x - pad, y - pad, x + total_width + pad, y + max_height + pad)
        overlay_draw.rectangle(rect, fill=(bg_color[0], bg_color[1], bg_color[2], int(255 * alpha)))

        # Draw the text segments on top in white (or specified text_color)
        cursor_x = x
        for seg_type, seg_text in segments:
            font = latin_font if seg_type == "latin" else cjk_font
            drawer.text((cursor_x, y), seg_text, font=font, fill=text_color)
            if has_textbbox:
                l, t, r, b = drawer.textbbox((cursor_x, y), seg_text, font=font)
                cursor_x = r
            else:
                seg_w, _ = drawer.textsize(seg_text, font=font)
                cursor_x += seg_w

    # Composite overlay onto image
    pil_image = Image.alpha_composite(pil_image, overlay)
    out_image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out_image


def _sanitize_filename(path: Path) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", path.stem)
    slug = slug.strip("_") or "image"
    return slug


def process_image(image_path: Path, recognizer: PCASVMRecognizer, detector: HaarFaceDetector, index: int) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    detections = detector.detect(image)
    annotated = image.copy()
    summary: dict[str, object] = {
        "file": image_path.name,
        "faces_detected": len(detections),
        "predictions": [],
    }

    h, w = annotated.shape[:2]
    min_edge = min(h, w)
    box_thickness = max(4, int(round(min_edge * 0.005)))
    font_size = max(28, int(round(min_edge * 0.035)))
    text_annotations: list[tuple[int, int, str, tuple[int, int, int], tuple[int, int, int], float]] = []

    if not detections:
        text_x = int(0.04 * w)
        text_y = int(0.08 * h)
        # Crimson background, white text
        text_annotations.append((text_x, text_y, "未检测到人脸", WHITE_RGB, CRIMSON_RGB, 0.6))
    else:
        face_tensors = [_prepare_face_patch(image, det.to_tuple()) for det in detections]
        X = np.vstack(face_tensors)
        predictions = recognizer.predict(X)

        for det, pred in zip(detections, predictions):
            x, y, w_box, h_box = det.to_tuple()
            # Gold bounding box
            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), GOLD_BGR, box_thickness)
            label_text = f"{pred.label} ({pred.confidence * 100:.1f}%)"
            text_y = max(y - int(0.35 * font_size), int(0.05 * h))
            # Semi-transparent gold background with white text
            text_annotations.append((x, text_y, label_text, WHITE_RGB, GOLD_RGB, 0.6))
            summary_entry = {
                "bbox": det.to_tuple(),
                "label": pred.label,
                "confidence": pred.confidence,
            }
            summary["predictions"].append(summary_entry)

    annotated = _render_text_annotations(annotated, text_annotations, font_size)

    slug = _sanitize_filename(image_path)
    output_name = f"{index:02d}_{slug}{image_path.suffix.lower()}"
    output_path = RESULTS_DIR / output_name
    cv2.imwrite(str(output_path), annotated)
    summary["output"] = str(output_path.relative_to(ROOT))
    summary["original_name"] = image_path.name
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run challenge set face detection and recognition demo")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help="Directory containing challenge images")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / MODEL_FILE, help="Path to recognizer model")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"识别模型不存在：{args.model}")

    if not args.source.exists():
        raise FileNotFoundError(f"挑战集目录不存在：{args.source}")

    recognizer = PCASVMRecognizer.load(args.model)
    detector = HaarFaceDetector()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    image_files = sorted([p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_files:
        raise RuntimeError("挑战集目录中未找到图片文件。")

    for idx, image_path in enumerate(image_files, start=1):
        summary = process_image(image_path, recognizer, detector, idx)
        summaries.append(summary)
        print(
            f"Processed {image_path.name}: faces={summary['faces_detected']}, "
            f"predictions={len(summary['predictions'])}"
        )

    report_path = RESULTS_DIR / "challenge_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("结果已保存：", report_path)


if __name__ == "__main__":
    main()
