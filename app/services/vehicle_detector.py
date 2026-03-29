import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import yaml
import os

logger = logging.getLogger(__name__)

class VehicleDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the vehicle detector with YOLOv8 model."""
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.class_names = self.config['model']['classes']
        self.confidence_threshold = self.config['model']['confidence_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model from weights file."""
        try:
            model_path = Path(self.config['model']['weights'])
            
            # If model doesn't exist in specified path, try to download it
            if not model_path.exists():
                logger.info(f"Model not found at {model_path}, attempting to download...")
                model_dir = model_path.parent
                os.makedirs(model_dir, exist_ok=True)
                
                # Download the model
                model = YOLO('yolov8x.pt')  # This will download if not present
                model.save(str(model_path))
                logger.info(f"Model downloaded and saved to {model_path}")
            else:
                model = YOLO(str(model_path))
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a frame and return their information.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of dictionaries containing detection information
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return []
            
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               classes=self.class_names)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_emergency': self._is_emergency_vehicle(class_name)
                    }
                    detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Error in vehicle detection: {str(e)}")
            return []
    
    def _is_emergency_vehicle(self, class_name: str) -> bool:
        """Check if the detected vehicle is an emergency vehicle."""
        emergency_classes = ['ambulance', 'fire_truck', 'police_car']
        return class_name in emergency_classes
    
    def get_vehicle_priority(self, detection: Dict) -> int:
        """
        Assign priority to detected vehicles.
        
        Priority levels:
        1: Emergency vehicles
        2: Buses
        3: Trucks
        4: Cars
        """
        if detection['is_emergency']:
            return 1
        elif detection['class_name'] == 'bus':
            return 2
        elif detection['class_name'] == 'truck':
            return 3
        else:
            return 4
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame and return annotated frame with detections.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return frame, []
            
            detections = self.detect_vehicles(frame)
            annotated_frame = frame.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                color = (0, 255, 0) if detection['is_emergency'] else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return annotated_frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, [] 