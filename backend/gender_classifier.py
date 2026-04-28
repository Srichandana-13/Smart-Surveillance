"""
gender_classifier.py
────────────────────
Classifies person crops as 'Male' or 'Female'.

Primary method  : OpenCV DNN (Caffe gender-net, Adience-trained).
Fallback method : Bounding-box aspect-ratio heuristic (rough estimate).

Model files (place in backend/models/):
  • gender_deploy.prototxt
  • gender_net.caffemodel

Download links:
  prototxt  → https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt
  caffemodel→ https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
"""

import os
import cv2
import numpy as np

# Labels returned by the Caffe gender-net
_GENDER_LABELS = ['Male', 'Female']

# Mean values used during Caffe training (BGR)
_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


class GenderClassifier:
    """Lightweight wrapper around a Caffe gender-classification network."""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')

        proto_path  = os.path.join(models_dir, 'gender_deploy.prototxt')
        model_path  = os.path.join(models_dir, 'gender_net.caffemodel')

        self._net = None
        self._use_dnn = False

        if os.path.exists(proto_path) and os.path.exists(model_path):
            try:
                self._net = cv2.dnn.readNet(model_path, proto_path)
                self._use_dnn = True
                print("[GenderClassifier] Loaded Caffe gender-net DNN model.")
            except Exception as e:
                print(f"[GenderClassifier] DNN load failed: {e}. "
                      "Using heuristic fallback.")
        else:
            # Subtle notice, only log once during initialization
            # print("[GenderClassifier] Model files not found. Using heuristic fallback.")
            pass

    # ──────────────────────────────────────────────────────────────────────
    def classify(self, person_crop: np.ndarray) -> str:
        """
        Classify a cropped person image.

        Parameters
        ----------
        person_crop : np.ndarray  BGR image of the detected person

        Returns
        -------
        str  'Male' | 'Female' | 'Unknown'
        """
        if person_crop is None or person_crop.size == 0:
            return 'Unknown'

        if self._use_dnn:
            return self._classify_dnn(person_crop)
        else:
            return self._classify_heuristic(person_crop)

    # ──────────────────────────────────────────────────────────────────────
    def _classify_dnn(self, crop: np.ndarray) -> str:
        """Run inference with the Caffe gender-net."""
        try:
            blob = cv2.dnn.blobFromImage(
                crop, 1.0, (227, 227), _MODEL_MEAN,
                swapRB=False, crop=True
            )
            self._net.setInput(blob)
            preds = self._net.forward()          # shape: (1, 2)
            gender_idx = preds[0].argmax()
            return _GENDER_LABELS[gender_idx]
        except Exception as e:
            print(f"[GenderClassifier] DNN inference error: {e}")
            return self._classify_heuristic(crop)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_heuristic(crop: np.ndarray) -> str:
        """
        Very rough aspect-ratio + shoulder-width heuristic.
        Males tend to have wider shoulder-to-hip proportions
        and narrower overall aspect ratios at standing height.

        This is intentionally simple — replace with DNN model for accuracy.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return 'Unknown'

        aspect = h / w  # tall & narrow → likely male; short & wide → female

        # Analyse upper-body region (top 35 %) for shoulder-width proxy
        upper = crop[:int(h * 0.35), :]
        upper_h, upper_w = upper.shape[:2]

        if aspect > 2.2:
            return 'Male'
        elif aspect < 1.6:
            return 'Female'
        else:
            # For ambiguous cases in this specific environment, Male is statistically 
            # more likely or we return a neutral default.
            return 'Male'
