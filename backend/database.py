import sqlite3
import os
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, 'database', 'surveillance.db')
        else:
            self.db_path = db_path
        self._init_db()
        self._migrate_db()

    # ─────────────────────────────────────────────────────────────────────
    def _init_db(self):
        """Create tables if they do not already exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ── Main detections table ────────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                object_type    TEXT    NOT NULL,
                event_type     TEXT    DEFAULT 'Entry',
                track_id       INTEGER,
                plate_number   TEXT,
                plate_img_path TEXT,
                image_path     TEXT    NOT NULL,
                camera_name    TEXT    DEFAULT 'Main',
                speed_kmh      REAL,
                gender         TEXT
            )
        ''')

        # ── PPE Violations table ─────────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppe_violations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                object_type    TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        # ── Seatbelt Violations table ────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seatbelt_violations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                vehicle_type   TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        conn.commit()

        # ── Mobile Usage table ───────────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mobile_usage (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                vehicle_type   TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        conn.commit()

        # ── Mobile Walking table ─────────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mobile_walking_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        conn.commit()

        # ── Restricted Zone table ────────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restricted_zone_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        conn.commit()

        # ── Sleeping Detection table ─────────────────────────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_detection_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                camera_name    TEXT    NOT NULL,
                violation_type TEXT    NOT NULL,
                image_path     TEXT    NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────────
    def _migrate_db(self):
        """Add new columns to existing databases (non-destructive migration)."""
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(detections)")
        columns = [col[1] for col in cursor.fetchall()]

        migrations = {
            'event_type':     "ALTER TABLE detections ADD COLUMN event_type     TEXT DEFAULT 'Entry'",
            'track_id':       "ALTER TABLE detections ADD COLUMN track_id       INTEGER",
            'plate_number':   "ALTER TABLE detections ADD COLUMN plate_number   TEXT",
            'plate_img_path': "ALTER TABLE detections ADD COLUMN plate_img_path TEXT",
            'camera_name':    "ALTER TABLE detections ADD COLUMN camera_name    TEXT DEFAULT 'Main'",
            'speed_kmh':      "ALTER TABLE detections ADD COLUMN speed_kmh      REAL",
            'gender':         "ALTER TABLE detections ADD COLUMN gender         TEXT",
        }

        for col, sql in migrations.items():
            if col not in columns:
                cursor.execute(sql)

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────────
    def insert_detection(self, object_type, image_path,
                         event_type='Entry', track_id=None,
                         plate_number=None, camera_name='Main',
                         speed_kmh=None, gender=None,
                         plate_img_path=None):
        """Persist one detection event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections
                (date, time, object_type, event_type, track_id,
                 plate_number, plate_img_path, image_path,
                 camera_name, speed_kmh, gender)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, object_type, event_type, track_id,
              plate_number, plate_img_path, image_path,
              camera_name, speed_kmh, gender))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def insert_night_alert(self, object_type, image_path,
                           plate_number=None, camera_name='Main'):
        """Persist one night-alert event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO night_alerts
                (date, time, object_type, plate_number, image_path, camera_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, object_type, plate_number,
              image_path, camera_name))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def insert_ppe_violation(self, object_type, violation_type, image_path,
                             camera_name='Main'):
        """Persist one PPE violation event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ppe_violations
                (date, time, camera_name, object_type, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, object_type,
              violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_ppe_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM ppe_violations ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_all_ppe_violations(self, date=None, camera=None):
        """Retrieve all PPE violations with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM ppe_violations WHERE 1=1"
        params = []
        
        if date:
            query += " AND date = ?"
            params.append(date)
        if camera:
            query += " AND camera_name = ?"
            params.append(camera)
            
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def get_ppe_stats_today(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM ppe_violations WHERE date=?", (date_str,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # ─────────────────────────────────────────────────────────────────────
    def insert_seatbelt_violation(self, vehicle_type, violation_type, image_path,
                                  camera_name='Main'):
        """Persist one seatbelt violation event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO seatbelt_violations
                (date, time, camera_name, vehicle_type, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, vehicle_type,
              violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_seatbelt_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM seatbelt_violations ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def insert_mobile_usage_violation(self, vehicle_type, violation_type, image_path,
                                      camera_name='Main'):
        """Persist one mobile usage violation event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO mobile_usage
                (date, time, camera_name, vehicle_type, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, vehicle_type,
              violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_mobile_usage_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM mobile_usage ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def insert_mobile_walking_violation(self, violation_type, image_path,
                                        camera_name='Main'):
        """Persist one pedestrian mobile usage violation event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO mobile_walking_logs
                (date, time, camera_name, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_mobile_walking_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM mobile_walking_logs ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def insert_restricted_zone_violation(self, violation_type, image_path,
                                         camera_name='Main'):
        """Persist one restricted zone entry event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO restricted_zone_logs
                (date, time, camera_name, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_restricted_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM restricted_zone_logs ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def insert_sleep_violation(self, violation_type, image_path,
                               camera_name='Main'):
        """Persist one sleep detection event."""
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sleep_detection_logs
                (date, time, camera_name, violation_type, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, time_str, camera_name, violation_type, image_path))
        conn.commit()
        conn.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    def get_recent_sleep_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM sleep_detection_logs ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def get_all_detections(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY id DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_recent_night_alerts(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM night_alerts ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_detections_by_category(self, categories):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        placeholders = ', '.join(['?'] * len(categories))
        query = (f'SELECT * FROM detections '
                 f'WHERE object_type IN ({placeholders}) ORDER BY id DESC')
        cursor.execute(query, categories)
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def get_detections_by_camera(self, camera_name):
        """Return all detections for a specific camera, newest first."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM detections WHERE camera_name = ? ORDER BY id DESC',
            (camera_name,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def get_plate_detections(self):
        """Return all detections that contain a number plate."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM detections '
            'WHERE plate_number IS NOT NULL '
            'ORDER BY id DESC'
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ─────────────────────────────────────────────────────────────────────
    def get_stats_for_report(self, date_str):
        """Statistics for a specific date used in PDF report generation."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM detections WHERE date=? AND event_type='Entry'",
            (date_str,)
        )
        entries = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM detections WHERE date=? AND event_type='Exit'",
            (date_str,)
        )
        exits = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE date=? AND object_type IN ('car','bus','truck','motorcycle')",
            (date_str,)
        )
        vehicles = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(DISTINCT plate_number) FROM detections "
            "WHERE date=? AND plate_number IS NOT NULL",
            (date_str,)
        )
        unique_plates = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM night_alerts WHERE date=?", (date_str,)
        )
        night_alerts = cursor.fetchone()[0]

        cursor.execute("""
            SELECT substr(time, 1, 2) AS hour, COUNT(*) AS count
            FROM detections
            WHERE date = ?
            GROUP BY hour
        """, (date_str,))
        hourly_data = {row['hour']: row['count'] for row in cursor.fetchall()}

        conn.close()
        return {
            'entries':      entries,
            'exits':        exits,
            'vehicles':     vehicles,
            'unique_plates': unique_plates,
            'night_alerts': night_alerts,
            'hourly_data':  hourly_data,
        }

    # ─────────────────────────────────────────────────────────────────────
    def get_global_stats(self):
        """Aggregate statistics for the dashboard header cards."""
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM detections WHERE event_type='Entry'")
        entries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detections WHERE event_type='Exit'")
        exits = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE object_type IN ('car','bus','truck','motorcycle')"
        )
        vehicles = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM night_alerts")
        night_alerts = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM ppe_violations WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        helmet_violations = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM seatbelt_violations WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        seatbelt_violations = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM mobile_usage WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        mobile_violations = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM mobile_walking_logs WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        mobile_walking_violations = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM restricted_zone_logs WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        restricted_violations = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM sleep_detection_logs WHERE date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        )
        sleep_violations = cursor.fetchone()[0]

        conn.close()
        return {
            'entries':           entries,
            'exits':             exits,
            'vehicles':          vehicles,
            'night_alerts':      night_alerts,
            'helmet_violations': helmet_violations,
            'seatbelt_violations': seatbelt_violations,
            'mobile_violations': mobile_violations,
            'mobile_walking_violations': mobile_walking_violations,
            'restricted_violations': restricted_violations,
            'sleep_violations': sleep_violations,
        }

    # ─────────────────────────────────────────────────────────────────────
    def search_detections(self, date=None, time=None,
                          plate_number=None, camera_name=None,
                          object_type=None):
        """
        Flexible detection search.
        All parameters are optional and ANDed together.
        Passing no parameters returns all records (most recent first).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query  = "SELECT * FROM detections WHERE 1=1"
        params = []

        if date:
            query += " AND date = ?"
            params.append(date)
        if time:
            query += " AND time LIKE ?"
            params.append(f"%{time}%")
        if plate_number:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate_number}%")
        if camera_name:
            query += " AND camera_name = ?"
            params.append(camera_name)
        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)

        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def clear_all_history(self):
        """Truncate all history and violation tables."""
        tables = [
            'detections', 'ppe_violations', 'seatbelt_violations',
            'mobile_usage', 'mobile_walking_logs', 'restricted_zone_logs',
            'sleep_detection_logs', 'night_alerts'
        ]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.close()
        return True

    def delete_detection(self, detection_id):
        """Delete a single detection record by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        conn.commit()
        conn.close()
        return True
