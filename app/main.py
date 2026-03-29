import os
import logging
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import random  # For demo data

from services.camera_manager import CameraManager

# Create necessary directories first
def create_directories():
    """Create necessary directories for the application."""
    directories = ['logs', 'data/models', 'data/datasets', 'uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories before setting up logging
create_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Load configuration
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

try:
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    raise

# Initialize camera manager
camera_manager = CameraManager()

def generate_frames(camera_id):
    """Generate video frames for streaming."""
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render dashboard page."""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    """Render analytics page."""
    return render_template('analytics.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Video streaming route."""
    return Response(generate_frames(camera_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# API Endpoints
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get list of active cameras."""
    return jsonify(list(camera_manager.cameras.keys()))

@app.route('/api/cameras/<camera_id>', methods=['POST'])
def add_camera(camera_id):
    """Add a new camera stream."""
    source = request.json.get('source')
    if not source:
        return jsonify({'error': 'Camera source not provided'}), 400
    
    success = camera_manager.add_camera(camera_id, source)
    if success:
        return jsonify({'message': f'Camera {camera_id} added successfully'})
    return jsonify({'error': f'Failed to add camera {camera_id}'}), 500

@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    """Remove a camera stream."""
    success = camera_manager.remove_camera(camera_id)
    if success:
        return jsonify({'message': f'Camera {camera_id} removed successfully'})
    return jsonify({'error': f'Failed to remove camera {camera_id}'}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get real-time dashboard statistics."""
    # Demo data - replace with actual data from your system
    return jsonify({
        'total_vehicles': random.randint(50, 200),
        'emergency_vehicles': random.randint(1, 10),
        'traffic_density': random.randint(30, 90),
        'response_time': random.randint(1, 5)
    })

@app.route('/api/dashboard/priority-alerts', methods=['GET'])
def get_priority_alerts():
    """Get priority vehicle alerts."""
    # Demo data - replace with actual data from your system
    alerts = []
    emergency_types = ['Ambulance', 'Fire Truck', 'Police']
    locations = ['Main Street', 'Highway 101', 'Downtown', 'West Side']
    
    for _ in range(random.randint(1, 5)):
        alert = {
            'type': random.choice(emergency_types),
            'location': random.choice(locations),
            'priority': random.choice(['High', 'Medium', 'Low']),
            'status': random.choice(['In Progress', 'Completed', 'Pending'])
        }
        alerts.append(alert)
    
    return jsonify(alerts)

@app.route('/api/analytics/data', methods=['POST'])
def get_analytics_data():
    """Get analytics data for the specified time range."""
    data = request.json
    time_range = data.get('time_range', '24h')
    
    # Generate demo data based on time range
    if time_range == '24h':
        labels = [f"{i:02d}:00" for i in range(24)]
        values = [random.randint(20, 100) for _ in range(24)]
    elif time_range == '7d':
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        values = [random.randint(100, 500) for _ in range(7)]
    else:  # 30d
        labels = [f"Day {i+1}" for i in range(30)]
        values = [random.randint(100, 500) for _ in range(30)]
    
    return jsonify({
        'traffic_volume': {
            'labels': labels,
            'values': values
        },
        'response_times': [
            random.randint(2, 5),  # Ambulance
            random.randint(3, 6),  # Fire Truck
            random.randint(1, 4)   # Police
        ],
        'avg_response_time': random.randint(2, 5),
        'priority_success': random.randint(90, 99),
        'flow_efficiency': random.randint(80, 95),
        'system_accuracy': random.randint(95, 99),
        'priority_analysis': [
            {
                'type': 'Ambulance',
                'detections': random.randint(50, 200),
                'avg_response_time': random.randint(2, 5),
                'success_rate': random.randint(90, 99),
                'trend': random.randint(-10, 10)
            },
            {
                'type': 'Fire Truck',
                'detections': random.randint(20, 100),
                'avg_response_time': random.randint(3, 6),
                'success_rate': random.randint(85, 95),
                'trend': random.randint(-10, 10)
            },
            {
                'type': 'Police',
                'detections': random.randint(100, 300),
                'avg_response_time': random.randint(1, 4),
                'success_rate': random.randint(95, 99),
                'trend': random.randint(-10, 10)
            }
        ]
    })

def main():
    """Main application entry point."""
    try:
        # Start camera manager
        camera_manager.start()
        
        # Run Flask application
        app.run(
            host=config['app']['host'],
            port=config['app']['port'],
            debug=config['app']['debug']
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    finally:
        camera_manager.stop()

if __name__ == '__main__':
    main() 