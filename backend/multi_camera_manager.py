import cv2
import json
import threading
import time
import os
from detector import SurveillanceDetector


class MultiCameraManager:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(base_dir, 'config', 'cameras.json')
        else:
            self.config_path = config_path

        self.cameras   = {}   # name -> source string
        self.detectors = {}   # name -> SurveillanceDetector
        self.frames    = {}   # name -> latest processed frame
        self.status    = {}   # name -> "Online" | "Offline"
        self.counts    = {}   # name -> (in_count, out_count)
        self.lock      = threading.Lock()
        self.load_config()

    # ─────────────────────────────────────────────────────────────────────
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                self.cameras = json.load(f)
            print(f"[CameraManager] Loaded {len(self.cameras)} camera(s).")
        except Exception as e:
            print(f"[CameraManager] Error loading config: {e}")
            self.cameras = {"Primary": "0"}

    # ─────────────────────────────────────────────────────────────────────
    def start_all(self):
        # Ensure recordings directory exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rec_dir = os.path.join(base_dir, 'recordings')
        os.makedirs(rec_dir, exist_ok=True)

        for name, source in self.cameras.items():
            cam_source = int(source) if source.isdigit() else source
            thread = threading.Thread(
                target=self._camera_loop,
                args=(name, cam_source),
                daemon=True
            )
            thread.start()
            print(f"[CameraManager] Started thread for camera: {name}")

    # ─────────────────────────────────────────────────────────────────────
    def _camera_loop(self, name: str, source):
        """Per-camera capture + detection loop (runs in its own thread)."""
        # Create a detector instance scoped to this camera
        detector = SurveillanceDetector(camera_name=name)
        with self.lock:
            self.detectors[name] = detector

        # Recording setup
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rec_dir = os.path.join(base_dir, 'recordings')
        video_writer = None
        current_date = None

        # ── Demonstration Fallback ──────────────────────────────────────────
        # If it's the Room camera and the source is a placeholder, try webcam
        actual_source = source
        is_placeholder = isinstance(source, str) and "placeholder" in source
        
        if is_placeholder:
            if name == "Room":
                print(f"[CameraManager] Placeholder detected for {name}. Falling back to webcam (0) for demo.")
                actual_source = 0
            else:
                print(f"[CameraManager] {name} using placeholder source. Setting to 'Offline' and silencing connection noise.")
                with self.lock:
                    self.status[name] = "Offline"
                # For non-Room placeholders that aren't using webcam, we don't need to loop-retry
                return

        while True:
            cap = cv2.VideoCapture(actual_source)

            # Verification of source opening
            if not cap.isOpened():
                if is_placeholder and name != "Room":
                   # This should have been caught by the return above, 
                   # but just as a safety measure for other types of bad sources:
                   with self.lock:
                       self.status[name] = "Offline"
                   return
                
                print(f"[CameraManager] Could not open {name} (source: {actual_source}). Retrying in 10 s…")
                with self.lock:
                    self.status[name] = "Offline"
                time.sleep(10)
                continue

            with self.lock:
                self.status[name] = "Online"

            while True:
                # Read multiple frames to clear the buffer and get the VERY LATEST one
                for _ in range(3):
                    cap.grab()
                success, frame = cap.read()

                if not success:
                    print(f"[CameraManager] Lost connection: {name}. Reconnecting…")
                    with self.lock:
                        self.status[name] = "Offline"
                    break

                # Run detection (pass camera_name for role-specific logic)
                processed_frame, counts = detector.process_frame(
                    frame, camera_name=name
                )

                # ── Recording Logic ──────────────────────────────────────
                now = time.localtime()
                date_str = time.strftime("%Y-%m-%d", now)
                
                # Create/Rotate video writer daily
                if video_writer is None or date_str != current_date:
                    if video_writer is not None:
                        video_writer.release()
                    
                    current_date = date_str
                    time_str = time.strftime("%H-%M-%S", now)
                    filename = f"{name}_{date_str}_{time_str}.mp4"
                    filepath = os.path.join(rec_dir, filename)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = processed_frame.shape[:2]
                    video_writer = cv2.VideoWriter(filepath, fourcc, 10.0, (w, h))
                    print(f"[CameraManager] Started recording: {filename}")

                if video_writer is not None:
                    video_writer.write(processed_frame)

                with self.lock:
                    self.frames[name]  = processed_frame
                    self.counts[name]  = counts
                    self.status[name]  = "Online"

            if video_writer is not None:
                video_writer.release()
                video_writer = None

            cap.release()
            time.sleep(2)

    # ─────────────────────────────────────────────────────────────────────
    # Public accessors
    # ─────────────────────────────────────────────────────────────────────
    def get_frame(self, name: str):
        with self.lock:
            return self.frames.get(name)

    def get_status(self, name: str) -> str:
        with self.lock:
            return self.status.get(name, "Offline")

    def get_counts(self, name: str) -> tuple:
        """Returns (in_count, out_count)."""
        with self.lock:
            return self.counts.get(name, (0, 0))

    def get_gender_counts(self, name: str) -> tuple:
        """
        Returns (male_count, female_count) for the given camera.
        Only meaningful for the Room camera.
        """
        with self.lock:
            detector = self.detectors.get(name)
        if detector is None:
            return (0, 0)
        return detector.get_gender_counts()

    def get_all_camera_names(self) -> list:
        return list(self.cameras.keys())
