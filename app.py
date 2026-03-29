from flask import Flask, request, render_template, redirect, url_for, session
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add a secret key for session management

# Load the YOLO model
model = YOLO("yolov8x.pt")  # Ensure the yolov8x.pt model file is present

# Expanded and more specific vehicle classes
vehicle_classes = [
    "car", "motorcycle", "motorbike", "truck", "bus", "auto", "bicycle", 
    "ambulance", "fire truck", "police car", "emergency vehicle"
]

# Priority vehicle time multipliers and detection confidence boost
priority_vehicles = {
    "ambulance": 5,
    "fire truck": 5,
    "police car": 4,
    "emergency vehicle": 5
}

# Mapping of similar vehicle names to standardize detection
vehicle_name_mapping = {
    "emergency vehicle": "ambulance",
    "fire engine": "fire truck",
    "police vehicle": "police car"
}

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}. Check if file is a valid image format.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1280, 1280))
    return image_resized

def calculate_time(vehicle_counts):
    # Time calculation with priority vehicle multipliers
    time_per_vehicle = {
        "car": 1,
        "motorcycle": 1,
        "motorbike": 1,
        "truck": 2,
        "bus": 2,
        "auto": 1,
        "bicycle": 1,
        "ambulance": 5,
        "fire truck": 5,
        "police car": 4,
        "emergency vehicle": 5
    }
    
    total_time = 0
    for vehicle, count in vehicle_counts.items():
        # Apply multiplier for priority vehicles
        if vehicle in priority_vehicles:
            total_time += time_per_vehicle.get(vehicle, 0) * count * 2
        else:
            total_time += time_per_vehicle.get(vehicle, 0) * count
    
    return total_time

def detect_vehicles(image):
    start_time = time.time()
    
    # Use lower confidence for initial detection to catch more vehicles
    results = model(image, imgsz=1280, conf=0.4, iou=0.5)
    detection_time = time.time() - start_time
    
    # Initialize vehicle counts dictionary
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}
    
    # Process detection results
    for box, conf, cls in zip(results[0].boxes.xywh, results[0].boxes.conf, results[0].boxes.cls):
        class_name = results[0].names[int(cls)].lower()
        
        # Normalize and map similar vehicle names
        if class_name in vehicle_name_mapping:
            class_name = vehicle_name_mapping[class_name]
        
        # Combine similar vehicle types
        if class_name == "motorcycle":
            class_name = "motorbike"
        
        # Check if detected class is in our target vehicle classes
        if class_name in vehicle_counts:
            # Boost confidence for priority vehicles
            if class_name in priority_vehicles:
                # Apply additional confidence check for priority vehicles
                if conf > 0.5:  # Higher confidence threshold for priority vehicles
                    vehicle_counts[class_name] += 1
            else:
                vehicle_counts[class_name] += 1
    
    # Remove zero-count vehicles for cleaner display
    vehicle_counts = {k: v for k, v in vehicle_counts.items() if v > 0}
    
    # Calculate estimated time with priority vehicle consideration
    estimated_time = calculate_time(vehicle_counts)
    
    return vehicle_counts, results[0].plot(), detection_time, estimated_time

# Rest of the Flask routes remain the same as in the previous implementation
@app.route("/", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        uploaded_files = []
        for i in range(1, 5):  # Expect 4 images for 4 roads
            file = request.files.get(f"file{i}")
            if file and file.filename:
                # Ensure uploads directory exists
                os.makedirs("uploads", exist_ok=True)
                # Ensure results directory exists
                os.makedirs(os.path.join("static", "results"), exist_ok=True)
                
                # Get file extension and validate
                filename = file.filename.lower()
                if not filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    continue
                
                file_path = os.path.join("uploads", f"road_{i}.jpg")
                file.save(file_path)
                
                # Verify file was saved and is readable
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    uploaded_files.append(file_path)

        if not uploaded_files:
            return render_template("upload.html", error="No valid image files uploaded")

        results = []
        for i, file_path in enumerate(uploaded_files, start=1):
            try:
                image = preprocess_image(file_path)
                vehicle_counts, result_image, detection_time, estimated_time = detect_vehicles(image)

                total_count = sum(vehicle_counts.values())
                result_path = os.path.join("static", "results", f"result_{i}.jpg")
                cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

                results.append({
                    "vehicle_counts": vehicle_counts,
                    "result_image": result_path,
                    "detection_time": round(detection_time, 2),
                    "total_count": total_count,
                    "estimated_time": estimated_time
                })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
            finally:
                # Remove the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)

        if not results:
            return render_template("upload.html", error="Failed to process any images")

        # Sort results by total vehicle count in descending order
        results = sorted(results, key=lambda x: x['total_count'], reverse=True)
        
        # Store results in session
        session['detection_results'] = results
        
        return render_template("result.html", results=results)

    return render_template("upload.html")

@app.route("/output")
def output_page():
    # Retrieve results from session
    results = session.get('detection_results', [])
    return render_template("output.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)