import unittest
from unittest.mock import patch

from src.engine import FaceEngine


class TestFaceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FaceEngine()

    @patch.object(FaceEngine, "validate_image_file", autospec=True)
    def test_compare_images_uses_supported_extract_faces_signature_and_prominent_face(self, mock_validate):
        engine = self.engine

        face_a_small = {"face": object(), "facial_area": {"w": 10, "h": 10}}
        face_a_large = {"face": object(), "facial_area": {"w": 30, "h": 25}}
        face_b_small = {"face": object(), "facial_area": {"w": 12, "h": 12}}
        face_b_large = {"face": object(), "facial_area": {"w": 40, "h": 20}}

        def extract_faces_stub(
            img_path,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            expand_percentage=0,
            grayscale=False,
            anti_spoofing=False,
        ):
            if img_path == "image-a.jpg":
                return [face_a_small, face_a_large]
            if img_path == "image-b.jpg":
                return [face_b_small, face_b_large]
            raise AssertionError(f"Unexpected image path: {img_path}")

        def verify_stub(
            img1_path,
            img2_path,
            model_name="ArcFace",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
            expand_percentage=0,
            normalization="base",
            silent=False,
            threshold=None,
            anti_spoofing=False,
        ):
            self.assertIs(img1_path, face_a_large["face"])
            self.assertIs(img2_path, face_b_large["face"])
            return {"distance": 0.22, "threshold": 0.68}

        with patch("src.engine.DeepFace.extract_faces", side_effect=extract_faces_stub) as mock_extract, patch(
            "src.engine.DeepFace.verify", side_effect=verify_stub
        ) as mock_verify:
            result = engine.compare_images("image-a.jpg", "image-b.jpg")

        self.assertIsNone(result["error"])
        self.assertTrue(result["match"])
        self.assertGreater(result["score"], 80.0)
        self.assertEqual(mock_extract.call_count, 2)
        mock_verify.assert_called_once()
        mock_validate.assert_any_call(engine, "image-a.jpg")
        mock_validate.assert_any_call(engine, "image-b.jpg")


if __name__ == "__main__":
    unittest.main()
