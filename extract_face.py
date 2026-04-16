#!/usr/bin/env python3
"""
extract_face.py — Detects and crops the face from a driver's license photo.

Usage:
    python extract_face.py <input_image> [output_image]

Example:
    python extract_face.py license.jpg face.jpg
"""

import sys
import os
import urllib.request
import cv2
import numpy as np

# OpenCV DNN face detector model files
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_PATH = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")

CONFIDENCE_THRESHOLD = 0.5
PADDING_RATIO = 0.175  # fraction of face size to add as padding on each side


def download_models():
    if not os.path.exists(PROTOTXT_PATH):
        print("Downloading face detector config...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)

    if not os.path.exists(MODEL_PATH):
        print("Downloading face detector model (~2MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def extract_face(input_path: str, output_path: str, padding: float = PADDING_RATIO):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    h, w = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    # The model expects a 300x300 blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    # Find the highest-confidence detection
    best = None
    best_confidence = 0.0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD and confidence > best_confidence:
            best_confidence = confidence
            best = detections[0, 0, i, 3:7]

    if best is None:
        raise RuntimeError("No face detected in the image.")

    # Scale bounding box back to original image size
    x1, y1, x2, y2 = (best * np.array([w, h, w, h])).astype(int)
    face_w, face_h = x2 - x1, y2 - y1

    # Apply padding, clamped to image bounds
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    face_crop = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, face_crop)

    print(f"Face detected (confidence: {best_confidence:.1%})")
    print(f"Saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "face.jpg"

    download_models()
    extract_face(input_path, output_path)


if __name__ == "__main__":
    main()
