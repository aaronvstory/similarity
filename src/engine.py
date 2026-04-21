import os
import threading
from typing import Dict, Union, Any, Optional
import urllib.request

import cv2
import numpy as np
from PIL import Image
# Ensure tf-keras backend is configured before importing deepface if possible.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace


class FaceEngine:
    """
    Singleton Backend for Face Detection and Recognition.
    Handles the initialization of ML models and the processing
    of similarity comparisons using DeepFace (ArcFace & RetinaFace).
    Also includes a fast OpenCV DNN-based face extraction module.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.model_name = "ArcFace"
        self.detector_backend = "retinaface"
        self.distance_metric = "cosine"
        
        # Extraction Model Paths (stored in src/models to keep root clean)
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.prototxt_path = os.path.join(self.models_dir, "deploy.prototxt")
        self.caffemodel_path = os.path.join(self.models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # URLs for extraction models
        self.prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        self.extraction_net = None
        self._initialized = True

    def initialize_models(self) -> None:
        """
        Pre-load the heavy ML models into memory. 
        This is typically called in a background thread upon app startup.
        """
        try:
            # Building the ArcFace model explicitly caches it into memory
            DeepFace.build_model(model_name=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Models: {e}")

    def _ensure_extraction_models(self) -> None:
        """Downloads the OpenCV DNN face detection models if missing."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.prototxt_path):
            urllib.request.urlretrieve(self.prototxt_url, self.prototxt_path)

        if not os.path.exists(self.caffemodel_path):
            urllib.request.urlretrieve(self.caffemodel_url, self.caffemodel_path)

        if self.extraction_net is None:
            self.extraction_net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)

    def extract_face(self, input_path: str, output_path: str, padding: float = 0.175) -> float:
        """
        Fast face extraction using OpenCV DNN module.
        Returns the confidence score of the detection.
        """
        self._ensure_extraction_models()
        
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        h, w = image.shape[:2]

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.extraction_net.setInput(blob)
        detections = self.extraction_net.forward()

        best = None
        best_confidence = 0.0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_confidence:
                best_confidence = confidence
                best = detections[0, 0, i, 3:7]

        if best is None:
            raise RuntimeError("No face detected in the image.")

        # Scale and crop
        x1, y1, x2, y2 = (best * np.array([w, h, w, h])).astype(int)
        face_w, face_h = x2 - x1, y2 - y1

        # Apply padding
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face_crop = image[y1:y2, x1:x2]
        cv2.imwrite(output_path, face_crop)
        
        return float(best_confidence)

    def validate_image_file(self, image_path: str) -> None:
        """
        Check if the file exists and is a valid image using PIL and cv2.
        Raises ValueError if corrupt or not found.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {image_path} ({e})")
            
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Unable to read image data via OpenCV: {image_path}")

    def _face_area(self, face_data: Dict[str, Any]) -> int:
        facial_area = face_data.get("facial_area", {}) if isinstance(face_data, dict) else {}
        width = facial_area.get("w", 0)
        height = facial_area.get("h", 0)
        try:
            return int(max(0, width) * max(0, height))
        except Exception:
            return 0

    def _select_prominent_face(self, faces: Any, image_label: str) -> Any:
        if not faces:
            raise ValueError(f"No face detected in {image_label}.")
        return max(faces, key=self._face_area)

    def compare_images(self, img1_path: str, img2_path: str) -> Dict[str, Union[bool, float, str]]:
        """
        Compares two images using the configured ML models.
        
        Returns:
            dict: {
                "match": bool (True if >= 80% score),
                "score": float (0-100 percentage),
                "error": str or None (Error message if something went wrong)
            }
        """
        try:
            # 1. Validate Files
            self.validate_image_file(img1_path)
            self.validate_image_file(img2_path)

            # 2. Extract faces explicitly first to catch "multiple faces" or "no faces" easily.
            # enforce_detection=True throws ValueError if no face is found.
            faces1 = DeepFace.extract_faces(
                img_path=img1_path, 
                detector_backend=self.detector_backend, 
                enforce_detection=True
            )
            faces2 = DeepFace.extract_faces(
                img_path=img2_path, 
                detector_backend=self.detector_backend, 
                enforce_detection=True
            )

            face1 = self._select_prominent_face(faces1, "image 1")
            face2 = self._select_prominent_face(faces2, "image 2")

            # 3. Perform Verification
            result = DeepFace.verify(
                img1_path=face1["face"],
                img2_path=face2["face"],
                model_name=self.model_name,
                detector_backend="skip",
                distance_metric=self.distance_metric,
                enforce_detection=False,
                align=True
            )

            # 4. Calculate Percentage using Threshold Mapping
            distance = float(result.get("distance", 1.0))
            threshold = float(result.get("threshold", 0.68))
            
            # Bound distance to [0, 1] just in case
            distance = max(0.0, min(1.0, distance))
            
            if distance <= threshold:
                # It IS a match! 
                # Map the distance [0.0 to threshold] to a score of [100% to 80%]
                similarity_score = 100.0 - ((distance / threshold) * 20.0)
                is_match = True
            else:
                # NOT a match. 
                # Map the distance [threshold to 1.0] to a score of [79% to 0%]
                similarity_score = max(0.0, 79.0 - (((distance - threshold) / (1.0 - threshold)) * 79.0))
                is_match = False

            return {
                "match": is_match,
                "score": round(similarity_score, 2),
                "error": None
            }

        except FileNotFoundError as e:
            return {"match": False, "score": 0.0, "error": str(e)}
        except ValueError as e:
            error_msg = str(e).lower()
            if "face could not be detected" in error_msg:
                return {"match": False, "score": 0.0, "error": "No face detected in one or both images. Ensure faces are clearly visible."}
            return {"match": False, "score": 0.0, "error": f"Validation Error: {e}"}
        except MemoryError:
            return {"match": False, "score": 0.0, "error": "Memory allocation error. The system ran out of RAM during processing."}
        except Exception as e:
            # Catch tf.errors.ResourceExhaustedError or similar indirectly
            error_msg = str(e).lower()
            if "exhausted" in error_msg or "oom" in error_msg or "memory" in error_msg:
                 return {"match": False, "score": 0.0, "error": "Memory resource exhausted. Please free up RAM."}
            return {"match": False, "score": 0.0, "error": f"An unexpected ML error occurred: {e}"}

