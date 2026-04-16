import os
import threading
from typing import Dict, Union, Any

import cv2
from PIL import Image
# Ensure tf-keras backend is configured before importing deepface if possible.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace


class FaceEngine:
    """
    Singleton Backend for Face Detection and Recognition.
    Handles the initialization of ML models and the processing
    of similarity comparisons using DeepFace (ArcFace & RetinaFace).
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

            if len(faces1) > 1:
                return {"match": False, "score": 0.0, "error": f"Multiple faces detected in image 1. Please use a photo with only one clearly visible face."}
            if len(faces2) > 1:
                return {"match": False, "score": 0.0, "error": f"Multiple faces detected in image 2. Please use a photo with only one clearly visible face."}

            # 3. Perform Verification
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=True,
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

