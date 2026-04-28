import cv2
import os
import time
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from tracker import ObjectTracker
from counter import PeopleCounter
from plate_reader import PlateReader
from database import DatabaseManager
from night_detector import NightDetector
from speed_estimator import SpeedEstimator
from gender_classifier import GenderClassifier
from helmet_detector import HelmetDetector
from seatbelt_detector import SeatbeltDetector
from mobile_detector import MobileDetector


# ─────────────────────────────────────────────────────────────────────────────
# Camera role constants – matched against camera names from cameras.json
# ─────────────────────────────────────────────────────────────────────────────
GATE_CAMERA    = 'Gate'
ROOM_CAMERA    = 'Room'
PARKING_CAMERA = 'Parking'


class SurveillanceDetector:
    def __init__(self, model_path=None, camera_name='Main'):
        # Base directory (backend/)
        self.base_dir    = os.path.dirname(os.path.abspath(__file__))
        self.camera_name = camera_name

        if model_path is None:
            model_path = os.path.join(self.base_dir, 'yolov8n.pt')

        # Core AI components
        self.model         = YOLO(model_path)
        self.tracker       = ObjectTracker()
        self.counter       = PeopleCounter(line_position=0.6)
        self.plate_reader  = PlateReader()
        self.db            = DatabaseManager()
        self.night_detector = NightDetector()

        # ── Change 1: Speed Estimator (Gate only) ────────────────────────
        self.speed_estimator = SpeedEstimator()

        # ── Change 2: Gender Classifier (Room only) ───────────────────────
        self.gender_classifier = GenderClassifier()
        self.male_count   = 0
        self.female_count = 0

        # ── Change 3: Helmet Detector (PPE Compliance) ────────────────────
        self.helmet_detector = HelmetDetector()
        self._ppe_violations_cache = {}  # track_id -> bool (has_violated)

        # ── Change 4: Seatbelt Detector ──────────────────────────────────
        self.seatbelt_detector = SeatbeltDetector()
        self._seatbelt_violations_cache = {} # track_id -> bool

        # ── Change 5: Mobile Usage Detector ──────────────────────────────
        self.mobile_detector = MobileDetector()
        self._mobile_violations_cache = {} # track_id -> bool
        self._mobile_walking_violations_cache = {} # track_id -> bool
        self._restricted_violations_cache = {} # track_id -> bool
        self._sleep_tracking = {} # tid -> start_time
        self._sleep_violations_cache = {} # tid -> bool

        # ── Change 6: Restricted Zones ──────────────────────────────────
        self.restricted_zones = self._load_restricted_zones()
        self.target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # Per-frame live speed overlay { track_id: speed_kmh }
        self._live_speeds: dict  = {}
        self._gender_history: dict = {}  # track_id -> list of 'Male'/'Female'
        self._gender_cache: dict = {}    # track_id -> stable gender string

        # Log paths
        self.log_file   = os.path.join(self.base_dir, 'logs', 'detections.log')
        self.system_log = os.path.join(self.base_dir, 'logs', 'system.log')

        for path in ['logs', 'images/persons', 'images/vehicles', 'images/plates', 'images/ppe', 'images/seatbelt', 'images/mobile_usage', 'images/mobile_walking', 'images/restricted', 'images/sleep']:
            full_path = os.path.join(self.base_dir, path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

    # ─────────────────────────────────────────────────────────────────────
    def log_system_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.system_log, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    # ─────────────────────────────────────────────────────────────────────
    def save_image(self, frame, obj_type, sub_dir,
                   x1=None, y1=None, x2=None, y2=None) -> str:
        """Save detected object image. Returns relative path for DB storage."""
        dir_path = os.path.join(self.base_dir, "images", sub_dir)
        os.makedirs(dir_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename  = f"{obj_type}_{timestamp}.jpg"
        filepath  = os.path.join(dir_path, filename)
        rel_path  = f"images/{sub_dir}/{filename}"

        if x1 is not None and y1 is not None:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(filepath, crop if crop.size > 0 else frame)
        else:
            cv2.imwrite(filepath, frame)

        return rel_path

    def _load_restricted_zones(self):
        """Load restricted zones from JSON config."""
        path = os.path.join(self.base_dir, 'config', 'restricted_zones.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _is_in_zone(self, point, polygon, frame_shape):
        """Check if a normalized point is inside a pixel-coordinate polygon."""
        h, w = frame_shape[:2]
        pixel_poly = np.array([[int(p[0]*w), int(p[1]*h)] for p in polygon], np.int32)
        dist = cv2.pointPolygonTest(pixel_poly, (int(point[0]), int(point[1])), False)
        return dist >= 0

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame, camera_name: str = None):
        """
        Process a single video frame.

        Returns: (annotated_frame, (in_count, out_count))
        """
        camera_name = camera_name or self.camera_name
        is_gate     = (camera_name == GATE_CAMERA)
        is_room     = (camera_name == ROOM_CAMERA)
        is_parking  = (camera_name == PARKING_CAMERA)

        # Use higher confidence for Room camera to avoid false positives (like chairs)
        # Use lower confidence for Gate/Parking to catch fast/distant vehicles
        conf_thresh = 0.45 if is_room else 0.25
        
        # run prediction with optimized thresholds
        results    = self.model(frame, verbose=False, conf=conf_thresh, iou=0.45)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) in self.target_classes:
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], score, int(class_id)))

        # Update tracker
        tracks = self.tracker.update(detections, frame)

        # Update crossing counter
        frame_h, frame_w, _ = frame.shape
        in_count, out_count, crossed_in, crossed_out = self.counter.count(
            tracks, frame_w, frame_h
        )

        # ── Change 1: Update speed estimates for gate / parking ───────────
        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            cx   = (ltrb[0] + ltrb[2]) / 2
            cy   = (ltrb[1] + ltrb[3]) / 2
            tid  = track.track_id
            active_ids.add(tid)

            if is_gate or is_parking:
                speed = self.speed_estimator.update(tid, cx, cy)
                self._live_speeds[tid] = speed

        # Cleanup logs for lost tracks
        self.speed_estimator.cleanup(active_ids)
        for tid in list(self._live_speeds.keys()):
            if tid not in active_ids:
                del self._live_speeds[tid]
        
        for tid in list(self._gender_cache.keys()):
            if tid not in active_ids:
                del self._gender_cache[tid]
                if tid in self._gender_history:
                    del self._gender_history[tid]

        # ── Handle crossing events ────────────────────────────────────────
        for track in crossed_in + crossed_out:
            event_type = 'Entry' if track in crossed_in else 'Exit'
            class_id   = track.get_det_class()
            obj_type   = self.target_classes.get(class_id, "unknown")
            ltrb       = track.to_ltrb()
            tid        = track.track_id

            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            # Ensure crop is within frame boundaries
            y1_c, y2_c = max(0, y1), min(frame_h, y2)
            x1_c, x2_c = max(0, x1), min(frame_w, x2)
            crop = frame[y1_c:y2_c, x1_c:x2_c]

            # ── Number-plate OCR: Gate AND Parking cameras ─────────────
            plate_number = None
            plate_img_path = None
            if obj_type in ['car', 'bus', 'truck', 'motorcycle']:
                if is_gate or is_parking:
                    if crop.size > 0:
                        plate_number = self.plate_reader.read_plate(crop)
                    if plate_number:
                        # Save a dedicated plate crop image
                        plate_img_path = self.save_image(
                            frame, f"plate_{obj_type}", "plates",
                            x1, y1, x2, y2
                        )
                        self.log_system_event(
                            f"PLATE: {plate_number} | {obj_type} | "
                            f"{event_type} | {camera_name}"
                        )

            # ── Speed (Gate / Parking) ────────────────────────────────
            speed_kmh = None
            if (is_gate or is_parking) and obj_type != 'person':
                speed_kmh = self._live_speeds.get(tid, 0.0)

            # ── Gender classification (Room camera – persons only) ────
            gender = None
            if is_room and obj_type == 'person':
                # Use the cached gender if available, else classify now
                gender = self._gender_cache.get(tid)
                if not gender and crop.size > 0:
                    gender = self.gender_classifier.classify(crop)
                
                # Update running male/female counts on crossing
                if event_type == 'Entry':
                    if gender == 'Male':
                        self.male_count += 1
                    elif gender == 'Female':
                        self.female_count += 1
                else: # Exit
                    # We can track exits if needed, but the user asked for 
                    # "how many males and females have come" (total) 
                    # and "how many are there in the room" (current).
                    # So we should probably track exit counts too.
                    if gender == 'Male':
                        self.male_count = max(0, self.male_count - 1)
                    elif gender == 'Female':
                        self.female_count = max(0, self.female_count - 1)

            # ── Save evidence image ───────────────────────────────────
            sub_dir   = "persons" if obj_type == 'person' else "vehicles"
            img_path  = self.save_image(frame, obj_type, sub_dir)

            # ── Night alert ───────────────────────────────────────────
            if self.night_detector.should_trigger_alert(obj_type):
                alert_img = self.save_image(
                    frame, f"night_{obj_type}", "vehicles",
                    x1, y1, x2, y2
                )
                self.db.insert_night_alert(
                    obj_type, alert_img, plate_number, camera_name
                )
                self.log_system_event(
                    f"NIGHT ALERT: {obj_type} at {camera_name}"
                )

            # ── Persist to DB ─────────────────────────────────────────
            self.db.insert_detection(
                object_type  = obj_type,
                image_path   = img_path,
                event_type   = event_type,
                track_id     = tid,
                plate_number = plate_number,
                camera_name  = camera_name,
                speed_kmh    = speed_kmh,
                gender       = gender,
                plate_img_path = plate_img_path,
            )
            # ── PPE / Helmet Detection ────────────────────────────────
            if obj_type == 'person':
                if crop.size > 0:
                    has_helmet, ppe_score, ppe_dets = self.helmet_detector.detect(crop)
                    
                    if has_helmet is False: # Violation
                        # Only record violation once per track to avoid spamming
                        if tid not in self._ppe_violations_cache:
                            violation_img = self.save_image(
                                frame, f"ppe_no_helmet", "ppe",
                                x1, y1, x2, y2
                            )
                            self.db.insert_ppe_violation(
                                object_type="person",
                                violation_type="No Helmet",
                                image_path=violation_img,
                                camera_name=camera_name
                            )
                            self._ppe_violations_cache[tid] = True
                            self.log_system_event(
                                f"PPE VIOLATION: No Helmet detected at {camera_name}"
                            )

                # ── Mobile Talking While Walking ──────────────────────────
                if crop.size > 0:
                    is_talking, muw_score, muw_dets = self.mobile_detector.detect_talking(crop)
                    if is_talking:
                        if tid not in self._mobile_walking_violations_cache:
                            violation_img = self.save_image(
                                frame, f"mobile_walking_violation", "mobile_walking",
                                x1, y1, x2, y2
                            )
                            self.db.insert_mobile_walking_violation(
                                violation_type="Unsafe Mobile Usage While Walking",
                                image_path=violation_img,
                                camera_name=camera_name
                            )
                            self._mobile_walking_violations_cache[tid] = True
                            self.log_system_event(
                                f"WALKING VIOLATION: Person talking on phone at {camera_name}"
                            )

                # ── Restricted Zone Detection ─────────────────────────────
                zones = self.restricted_zones.get(camera_name, [])
                # Use bottom-center of bounding box (feet position)
                feet_pos = [(x1 + x2) / 2, y2]
                
                for zone in zones:
                    if self._is_in_zone(feet_pos, zone['polygon'], frame.shape):
                        if tid not in self._restricted_violations_cache:
                            violation_img = self.save_image(
                                frame, f"restricted_entry", "restricted",
                                x1, y1, x2, y2
                            )
                            self.db.insert_restricted_zone_violation(
                                violation_type=f"Restricted Pathway Entry ({zone['name']})",
                                image_path=violation_img,
                                camera_name=camera_name
                            )
                            self._restricted_violations_cache[tid] = True
                            self.log_system_event(
                                f"ZONE VIOLATION: Restricted entry in {zone['name']} at {camera_name}"
                            )

                # ── Sleeping Detection ────────────────────────────────────
                # Horizontal orientation: width > height * 1.2 (heuristic)
                is_horizontal = (x2 - x1) > (y2 - y1) * 1.1
                
                if is_horizontal:
                    if tid not in self._sleep_tracking:
                        self._sleep_tracking[tid] = time.time()
                    else:
                        duration = time.time() - self._sleep_tracking[tid]
                        if duration > 120: # 2 minutes
                            if tid not in self._sleep_violations_cache:
                                violation_img = self.save_image(
                                    frame, f"sleeping_on_duty", "sleep",
                                    x1, y1, x2, y2
                                )
                                self.db.insert_sleep_violation(
                                    violation_type="Sleeping On Duty Alert",
                                    image_path=violation_img,
                                    camera_name=camera_name
                                )
                                self._sleep_violations_cache[tid] = True
                                self.log_system_event(
                                    f"SLEEPING VIOLATION: Person detected sleeping at {camera_name}"
                                )
                else:
                    # Reset if they stand up
                    if tid in self._sleep_tracking:
                        del self._sleep_tracking[tid]

            # ── Seatbelt Detection (4-wheelers) ───────────────────────
            if obj_type in ['car', 'bus', 'truck']:
                if crop.size > 0:
                    has_sb, sb_score, sb_dets = self.seatbelt_detector.detect(crop)
                    
                    if has_sb is False: # Violation
                        if tid not in self._seatbelt_violations_cache:
                            violation_img = self.save_image(
                                frame, f"seatbelt_violation", "seatbelt",
                                x1, y1, x2, y2
                            )
                            self.db.insert_seatbelt_violation(
                                vehicle_type=obj_type,
                                violation_type="No Seatbelt",
                                image_path=violation_img,
                                camera_name=camera_name
                            )
                            self._seatbelt_violations_cache[tid] = True
                            self.log_system_event(
                                f"SEATBELT VIOLATION: No seatbelt detected on {obj_type} at {camera_name}"
                            )

            # ── Mobile Usage Detection (4-wheelers) ───────────────────
            if obj_type in ['car', 'bus', 'truck']:
                if crop.size > 0:
                    is_using_phone, mu_score, mu_dets = self.mobile_detector.detect(crop)
                    
                    if is_using_phone:
                        if tid not in self._mobile_violations_cache:
                            violation_img = self.save_image(
                                frame, f"mobile_usage_violation", "mobile_usage",
                                x1, y1, x2, y2
                            )
                            self.db.insert_mobile_usage_violation(
                                vehicle_type=obj_type,
                                violation_type="Mobile Usage While Driving",
                                image_path=violation_img,
                                camera_name=camera_name
                            )
                            self._mobile_violations_cache[tid] = True
                            self.log_system_event(
                                f"MOBILE USAGE VIOLATION: Driver using phone on {obj_type} at {camera_name}"
                            )

            self.log_system_event(
                f"EVENT: {obj_type} ({event_type}) at {camera_name}"
                + (f" | {speed_kmh:.1f} km/h" if speed_kmh else "")
                + (f" | Plate: {plate_number}" if plate_number else "")
                + (f" | Gender: {gender}" if gender else "")
            )

        # ── Visualization ─────────────────────────────────────────────────
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb     = track.to_ltrb()
            obj_type = self.target_classes.get(track.get_det_class(), "unknown")
            tid      = track.track_id
            color    = (0, 255, 0) if obj_type == 'person' else (255, 165, 0)

            cv2.rectangle(
                frame,
                (int(ltrb[0]), int(ltrb[1])),
                (int(ltrb[2]), int(ltrb[3])),
                color, 2
            )

            # Build label
            label = f"{obj_type} #{tid}"

            # ── Append speed on Gate / Parking ─────────────────────
            if (is_gate or is_parking) and obj_type != 'person':
                spd = self._live_speeds.get(tid, 0.0)
                label += f"  {spd:.1f} km/h"

            # ── Stable Gender Classification (Majority Voting) ────────
            if is_room and obj_type == 'person':
                # Only update if not already "stabilized" to a high degree
                if tid not in self._gender_cache or len(self._gender_history.get(tid, [])) < 15:
                    ltrb2 = track.to_ltrb()
                    # Crop with a slight margin for better gender accuracy
                    margin = 10
                    y1_m = max(0, int(ltrb2[1]) - margin)
                    y2_m = min(frame.shape[0], int(ltrb2[3]) + margin)
                    x1_m = max(0, int(ltrb2[0]) - margin)
                    x2_m = min(frame.shape[1], int(ltrb2[2]) + margin)
                    
                    crop2 = frame[y1_m:y2_m, x1_m:x2_m]
                    if crop2.size > 0:
                        g = self.gender_classifier.classify(crop2)
                        
                        if tid not in self._gender_history:
                            self._gender_history[tid] = []
                        self._gender_history[tid].append(g)
                        
                        # Once we have 5+ readings, take the most common one
                        if len(self._gender_history[tid]) >= 5:
                            m_count = self._gender_history[tid].count('Male')
                            f_count = self._gender_history[tid].count('Female')
                            self._gender_cache[tid] = 'Male' if m_count >= f_count else 'Female'
                
                g = self._gender_cache.get(tid, "Analyzing...")
                label += f"  {g}"
                
                # Colour-code: Blue for Male, Magenta for Female, Gray for Analyzing
                if g == 'Male':
                    color = (255, 120, 0)
                elif g == 'Female':
                    color = (180, 0, 255)
                else:
                    color = (150, 150, 150)

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), 
                              (int(ltrb[2]), int(ltrb[3])), color, 2)

            # ── PPE Visual Indicator ──────────────────────────────
            if obj_type == 'person':
                ltrb_p = track.to_ltrb()
                y1_p, y2_p = max(0, int(ltrb_p[1])), min(frame.shape[0], int(ltrb_p[3]))
                x1_p, x2_p = max(0, int(ltrb_p[0])), min(frame.shape[1], int(ltrb_p[2]))
                crop_p = frame[y1_p:y2_p, x1_p:x2_p]
                
                if crop_p.size > 0:
                    has_h, _, _ = self.helmet_detector.detect(crop_p)
                    if has_h is True:
                        ppe_label = "SAFE (Helmet)"
                        ppe_color = (0, 255, 0)
                    elif has_h is False:
                        ppe_label = "ALERT (No Helmet)"
                        ppe_color = (0, 0, 255)
                    else:
                        ppe_label = "PPE: Checking..."
                        ppe_color = (255, 255, 255)
                    
                    cv2.putText(frame, ppe_label,
                                (int(ltrb_p[0]), int(ltrb_p[3]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ppe_color, 2)

                    # ── Mobile Walking Visual Indicator ───────────────
                    is_t, _, _ = self.mobile_detector.detect_talking(crop_p)
                    if is_t:
                        cv2.putText(frame, "UNSAFE: MOBILE TALKING",
                                    (int(ltrb_p[0]), int(ltrb_p[3]) + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ── Restricted Zone Boundaries ────────────────────────
            zones = self.restricted_zones.get(camera_name, [])
            for zone in zones:
                h, w = frame.shape[:2]
                pixel_poly = np.array([[int(p[0]*w), int(p[1]*h)] for p in zone['polygon']], np.int32)
                cv2.polylines(frame, [pixel_poly], True, (0, 0, 255), 2)
                cv2.putText(frame, f"RESTRICTED: {zone['name']}",
                            (pixel_poly[0][0], pixel_poly[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # ── Sleeping Visual Indicator ─────────────────────────
            if obj_type == 'person':
                if tid in self._sleep_tracking:
                    duration = time.time() - self._sleep_tracking[tid]
                    if duration > 10: # show status after 10s of lying down
                        color_s = (0, 0, 255) if duration > 120 else (0, 165, 255)
                        label_s = "SLEEPING!!" if duration > 120 else f"LYING DOWN: {int(duration)}s"
                        cv2.putText(frame, label_s,
                                    (int(ltrb[0]), int(ltrb[1]) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_s, 2)

            # ── Seatbelt Visual Indicator ─────────────────────────
            if obj_type in ['car', 'bus', 'truck']:
                ltrb_v = track.to_ltrb()
                y1_v, y2_v = max(0, int(ltrb_v[1])), min(frame.shape[0], int(ltrb_v[3]))
                x1_v, x2_v = max(0, int(ltrb_v[0])), min(frame.shape[1], int(ltrb_v[2]))
                crop_v = frame[y1_v:y2_v, x1_v:x2_v]
                
                if crop_v.size > 0:
                    has_s, _, _ = self.seatbelt_detector.detect(crop_v)
                    if has_s is True:
                        sb_label = "SEATBELT: OK"
                        sb_color = (0, 255, 0)
                    elif has_s is False:
                        sb_label = "SEATBELT: MISSING"
                        sb_color = (0, 0, 255)
                    else:
                        sb_label = "SEATBELT: CHECKING"
                        sb_color = (255, 255, 255)
                    
                    cv2.putText(frame, sb_label,
                                (int(ltrb_v[0]), int(ltrb_v[3]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, sb_color, 2)

            # ── Mobile Usage Visual Indicator ─────────────────────
            if obj_type in ['car', 'bus', 'truck']:
                ltrb_m = track.to_ltrb()
                y1_m2, y2_m2 = max(0, int(ltrb_m[1])), min(frame.shape[0], int(ltrb_m[3]))
                x1_m2, x2_m2 = max(0, int(ltrb_m[0])), min(frame.shape[1], int(ltrb_m[2]))
                crop_m = frame[y1_m2:y2_m2, x1_m2:x2_m2]
                
                if crop_m.size > 0:
                    using_p, _, _ = self.mobile_detector.detect(crop_m)
                    if using_p:
                        mu_label = "MOBILE: USING PHONE"
                        mu_color = (0, 0, 255)
                        cv2.putText(frame, mu_label,
                                    (int(ltrb_m[0]), int(ltrb_m[3]) + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mu_color, 2)

            cv2.putText(
                frame, label,
                (int(ltrb[0]), int(ltrb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # ── Entry / Exit boundary line ────────────────────────────────────
        line_y    = int(self.counter.line_position * frame_h)
        buffer_px = int(self.counter.buffer_size * frame_h / 2)

        cv2.line(frame, (0, line_y - buffer_px), (frame_w, line_y - buffer_px),
                 (100, 100, 100), 1)
        cv2.line(frame, (0, line_y + buffer_px), (frame_w, line_y + buffer_px),
                 (100, 100, 100), 1)
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 3)
        cv2.putText(frame, "ENTRY / EXIT BOUNDARY",
                    (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, (in_count, out_count)

    # ─────────────────────────────────────────────────────────────────────
    def get_gender_counts(self) -> tuple:
        """Return (male_count, female_count) for this detector instance."""
        return self.male_count, self.female_count
