from flask import (Flask, render_template, Response, request,
                   jsonify, send_from_directory, send_file, abort)
import cv2
import threading
import os
import json
import psutil
import numpy as np
from datetime import datetime

from multi_camera_manager import MultiCameraManager
from database import DatabaseManager
from report_generator import ReportGenerator

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(FRONTEND_DIR, 'static')
)

db             = DatabaseManager()
camera_manager = MultiCameraManager()
report_gen     = ReportGenerator(db)

# Start background processes
camera_manager.start_all()
report_gen.run_scheduler()


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    cameras = camera_manager.get_all_camera_names()
    return render_template('dashboard.html', cameras=cameras)


@app.route('/history')
def history():
    detections = db.get_all_detections()
    return render_template('history.html', detections=detections)


@app.route('/reports')
def reports_page():
    reports_dir = os.path.join(BASE_DIR, 'reports', 'daily_reports')
    os.makedirs(reports_dir, exist_ok=True)
    pdfs = sorted(
        [f for f in os.listdir(reports_dir) if f.endswith('.pdf')],
        reverse=True
    )
    return render_template('reports.html', report_files=pdfs)


@app.route('/humans')
def humans():
    detections = db.get_detections_by_category(['person'])
    return render_template('history.html',
                           title="HUMAN DETECTIONS", detections=detections)


@app.route('/bikes')
def bikes():
    detections = db.get_detections_by_category(['motorcycle', 'bicycle'])
    return render_template('history.html',
                           title="TWO-WHEELER DETECTIONS", detections=detections)


@app.route('/vehicles')
def vehicles():
    detections = db.get_detections_by_category(['car', 'bus', 'truck'])
    return render_template('history.html',
                           title="VEHICLE DETECTIONS", detections=detections)


@app.route('/search')
def search_page():
    """Render the search page (results loaded via JS + POST API)."""
    return render_template('search.html')


@app.route('/ppe-violations')
def ppe_violations_page():
    """PPE Violations history page."""
    # Fetch all camera names for filter
    cameras = camera_manager.get_all_camera_names()
    return render_template('ppe_violations.html', cameras=cameras)


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Accept a JSON body with optional filter fields:
      date, time, plate_number, camera_name, object_type
    Return matching detection rows as JSON.
    """
    body        = request.get_json(force=True, silent=True) or {}
    date        = body.get('date')        or None
    time_val    = body.get('time')        or None
    plate       = body.get('plate_number') or None
    camera      = body.get('camera_name') or None
    object_type = body.get('object_type') or None

    rows = db.search_detections(
        date=date,
        time=time_val,
        plate_number=plate,
        camera_name=camera,
        object_type=object_type,
    )
    return jsonify([dict(r) for r in rows])


@app.route('/plates')
def plates():
    """All detections that include a number plate (Gate + Parking cameras)."""
    detections = db.get_plate_detections()
    return render_template('history.html',
                           title="NUMBER PLATE DETECTIONS", detections=detections)


@app.route('/settings')
def settings_page():
    return render_template('settings.html')


@app.route('/cameras')
def cameras_page():
    return render_template('cameras.html')


@app.route('/logs')
def logs_page():
    return render_template('logs.html')


# ─────────────────────────────────────────────────────────────────────────────
# Video stream
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    def gen_frames():
        while True:
            frame = camera_manager.get_frame(camera_name)
            if frame is None:
                # Offline placeholder
                offline_path = os.path.join(FRONTEND_DIR,
                                            'static/images/offline.jpg')
                placeholder  = cv2.imread(offline_path)
                if placeholder is None:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, f"Camera {camera_name} Offline",
                        (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2
                    )
                ret, buffer = cv2.imencode('.jpg', placeholder)
            else:
                ret, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
            
            # Prevent extreme CPU looping
            import time
            time.sleep(0.03)

    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/camera_states')
def camera_states():
    """
    Returns per-camera runtime state including:
      - status, in/out counts
      - speed (Gate / Parking via DB latest record)
      - male / female counts (Room)
    """
    states = {}
    for name in camera_manager.get_all_camera_names():
        counts         = camera_manager.get_counts(name)
        gender_counts  = camera_manager.get_gender_counts(name)

        states[name] = {
            'status':       camera_manager.get_status(name),
            'in':           counts[0],
            'out':          counts[1],
            'male_count':   gender_counts[0],
            'female_count': gender_counts[1],
        }
    return jsonify(states)


@app.route('/api/night_alerts')
def night_alerts():
    alerts = db.get_recent_night_alerts(5)
    return jsonify([dict(row) for row in alerts])


@app.route('/api/ppe_violations')
def ppe_violations():
    violations = db.get_recent_ppe_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/ppe_violations_filtered', methods=['POST'])
def ppe_violations_filtered():
    """Returns filtered PPE violations."""
    body   = request.get_json(force=True, silent=True) or {}
    date   = body.get('date') or None
    camera = body.get('camera') or None
    
    rows = db.get_all_ppe_violations(date=date, camera=camera)
    return jsonify([dict(r) for r in rows])


@app.route('/api/seatbelt_violations')
def seatbelt_violations():
    violations = db.get_recent_seatbelt_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/mobile_violations')
def mobile_violations():
    violations = db.get_recent_mobile_usage_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/mobile_walking_violations')
def mobile_walking_violations():
    violations = db.get_recent_mobile_walking_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/restricted_violations')
def restricted_violations():
    violations = db.get_recent_restricted_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/sleep_violations')
def sleep_violations():
    violations = db.get_recent_sleep_violations(5)
    return jsonify([dict(row) for row in violations])


@app.route('/api/activity_feed')
def activity_feed():
    """Returns a unified feed of recent detections and violations."""
    # Get recent detections
    detections = db.get_all_detections()[:15]
    # Get recent PPE violations
    ppe = db.get_recent_ppe_violations(15)
    
    feed = []
    for d in detections:
        feed.append({
            'id': d['id'],
            'type': 'detection',
            'category': d['object_type'],
            'message': f"{d['object_type']} {d['event_type']} at {d['camera_name']}",
            'time': d['time'],
            'date': d['date'],
            'image': f"/{d['image_path']}",
            'camera': d['camera_name']
        })
    
    for v in ppe:
        feed.append({
            'id': v['id'],
            'type': 'violation',
            'category': 'PPE',
            'message': f"PPE VIOLATION: {v['violation_type']} at {v['camera_name']}",
            'time': v['time'],
            'date': v['date'],
            'image': f"/{v['image_path']}",
            'camera': v['camera_name']
        })
    
    # Sort by time/id descending
    feed.sort(key=lambda x: (x['date'], x['time']), reverse=True)
    return jsonify(feed[:15])


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    db.clear_all_history()
    return jsonify({"status": "success", "message": "All history cleared"})


@app.route('/api/detections/<int:det_id>', methods=['DELETE'])
def delete_detection(det_id):
    db.delete_detection(det_id)
    return jsonify({"status": "success", "message": f"Detection {det_id} deleted"})


@app.route('/api/stats')
def get_global_stats():
    stats = db.get_global_stats()
    return jsonify(stats)


@app.route('/api/system_health')
def system_health():
    """Return CPU, Memory and Disk usage."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        return jsonify({
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "status": "Healthy"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Read or update system settings."""
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.json')
    
    if request.method == 'POST':
        new_settings = request.json
        with open(settings_path, 'w') as f:
            json.dump(new_settings, f, indent=4)
        return jsonify({"status": "success"})
    
    # GET
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route('/api/camera_control', methods=['POST'])
def camera_control():
    """Simulate camera start/stop/restart."""
    data = request.json
    camera_id = data.get('camera')
    action = data.get('action') # start, stop, restart
    
    # In a real system, we would manage the CaptureThread here
    # For now, we simulate success
    return jsonify({
        "status": "success",
        "camera": camera_id,
        "action": action
    })


@app.route('/api/generate_report')
def generate_report():
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = report_gen.generate_daily_report(date_str)
    return send_file(filepath, as_attachment=True)


@app.route('/api/logs')
def get_logs():
    log_path = os.path.join(BASE_DIR, 'logs', 'system.log')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
        return jsonify([line.strip() for line in lines[-15:]])
    return jsonify([])


@app.route('/api/recent_detections/<camera_name>')
def recent_detections(camera_name):
    """
    Returns the last 10 detections for a given camera,
    including speed_kmh and gender fields.
    """
    rows = db.get_detections_by_camera(camera_name)[:10]
    result = []
    for row in rows:
        d = dict(row)
        result.append(d)
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# Static file helpers
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'images'), filename)


@app.route('/reports/daily_reports/<path:filename>')
def serve_report(filename):
    reports_dir = os.path.join(BASE_DIR, 'reports', 'daily_reports')
    return send_from_directory(reports_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
