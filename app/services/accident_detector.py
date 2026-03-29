import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class AccidentDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the accident detector with configuration."""
        self.config = self._load_config(config_path)
        self.motion_threshold = self.config['accident_detection']['motion_threshold']
        self.collision_threshold = self.config['accident_detection']['collision_threshold']
        self.alert_cooldown = self.config['accident_detection']['alert_cooldown']
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Initialize motion history
        self.motion_history = []
        self.last_alert_time = datetime.min
        self.accident_detected = False
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect motion in the frame using background subtraction.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (motion_score, motion_mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion score
        motion_score = np.sum(fg_mask) / (frame.shape[0] * frame.shape[1])
        
        return motion_score, fg_mask
    
    def detect_collision(self, detections: List[Dict]) -> bool:
        """
        Detect potential collisions between vehicles.
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Boolean indicating if collision is detected
        """
        if len(detections) < 2:
            return False
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if self._check_intersection(detections[i]['bbox'], detections[j]['bbox']):
                    return True
        return False
    
    def _check_intersection(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes intersect."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection over union
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        
        return iou > self.collision_threshold
    
    def process_frame(self, frame: np.ndarray, 
                     detections: List[Dict]) -> Tuple[np.ndarray, bool]:
        """
        Process a frame to detect accidents.
        
        Args:
            frame: Input frame
            detections: List of vehicle detections
            
        Returns:
            Tuple of (annotated_frame, accident_detected)
        """
        # Detect motion
        motion_score, motion_mask = self.detect_motion(frame)
        self.motion_history.append(motion_score)
        
        # Keep only recent motion history
        if len(self.motion_history) > 30:
            self.motion_history.pop(0)
        
        # Detect collision
        collision_detected = self.detect_collision(detections)
        
        # Check for accident conditions
        current_time = datetime.now()
        time_since_last_alert = (current_time - self.last_alert_time).total_seconds()
        
        if (motion_score > self.motion_threshold and 
            collision_detected and 
            time_since_last_alert > self.alert_cooldown):
            self.accident_detected = True
            self.last_alert_time = current_time
            logger.warning("Accident detected!")
        else:
            self.accident_detected = False
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Add motion visualization
        motion_overlay = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, motion_overlay, 0.3, 0)
        
        # Add accident status
        status_text = "ACCIDENT DETECTED!" if self.accident_detected else "No Accident"
        color = (0, 0, 255) if self.accident_detected else (0, 255, 0)
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return annotated_frame, self.accident_detected 