import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import yaml
from pathlib import Path

from .vehicle_detector import VehicleDetector
from .accident_detector import AccidentDetector

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the camera manager with configuration."""
        self.config = self._load_config(config_path)
        self.max_workers = self.config['cameras']['max_workers']
        self.frame_buffer_size = self.config['cameras']['frame_buffer_size']
        self.processing_interval = self.config['cameras']['processing_interval']
        
        # Initialize detectors
        self.vehicle_detector = VehicleDetector(config_path)
        self.accident_detector = AccidentDetector(config_path)
        
        # Initialize camera streams and processing queues
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_camera(self, camera_id: str, source: str) -> bool:
        """
        Add a new camera stream.
        
        Args:
            camera_id: Unique identifier for the camera
            source: Camera source (URL, RTSP stream, or device index)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id} at {source}")
                return False
            
            # Store camera and create frame queue
            self.cameras[camera_id] = cap
            self.frame_queues[camera_id] = queue.Queue(maxsize=self.frame_buffer_size)
            
            # Start processing thread
            self.processing_threads[camera_id] = threading.Thread(
                target=self._process_camera_stream,
                args=(camera_id,),
                daemon=True
            )
            self.processing_threads[camera_id].start()
            
            logger.info(f"Successfully added camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {str(e)}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera stream.
        
        Args:
            camera_id: Camera identifier to remove
            
        Returns:
            Boolean indicating success
        """
        try:
            if camera_id in self.cameras:
                # Stop processing thread
                if camera_id in self.processing_threads:
                    self.processing_threads[camera_id].join(timeout=1.0)
                    del self.processing_threads[camera_id]
                
                # Release camera
                self.cameras[camera_id].release()
                del self.cameras[camera_id]
                
                # Clear frame queue
                while not self.frame_queues[camera_id].empty():
                    self.frame_queues[camera_id].get_nowait()
                del self.frame_queues[camera_id]
                
                logger.info(f"Successfully removed camera {camera_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing camera {camera_id}: {str(e)}")
            return False
    
    def _process_camera_stream(self, camera_id: str):
        """Process frames from a camera stream."""
        while self.is_running and camera_id in self.cameras:
            try:
                ret, frame = self.cameras[camera_id].read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Add frame to queue
                if not self.frame_queues[camera_id].full():
                    self.frame_queues[camera_id].put(frame)
                
                # Process frame in thread pool
                self.executor.submit(self._process_frame, camera_id, frame)
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {str(e)}")
                time.sleep(0.1)
    
    def _process_frame(self, camera_id: str, frame: np.ndarray):
        """
        Process a single frame with vehicle and accident detection.
        
        Args:
            camera_id: Camera identifier
            frame: Input frame
        """
        try:
            # Detect vehicles
            annotated_frame, detections = self.vehicle_detector.process_frame(frame)
            
            # Detect accidents
            final_frame, accident_detected = self.accident_detector.process_frame(
                annotated_frame, detections
            )
            
            # Store results or trigger alerts
            if accident_detected:
                self._handle_accident(camera_id, detections)
            
            # Update frame in queue
            if not self.frame_queues[camera_id].full():
                self.frame_queues[camera_id].put(final_frame)
                
        except Exception as e:
            logger.error(f"Error in frame processing for camera {camera_id}: {str(e)}")
    
    def _handle_accident(self, camera_id: str, detections: List[Dict]):
        """Handle detected accident by triggering alerts."""
        try:
            # Get emergency service endpoint from config
            endpoint = self.config['emergency_services']['notification_endpoint']
            
            # Prepare accident data
            accident_data = {
                'camera_id': camera_id,
                'timestamp': time.time(),
                'detections': detections,
                'location': self._get_camera_location(camera_id)
            }
            
            # TODO: Implement emergency service notification
            logger.warning(f"Accident detected on camera {camera_id}")
            logger.info(f"Accident data: {accident_data}")
            
        except Exception as e:
            logger.error(f"Error handling accident for camera {camera_id}: {str(e)}")
    
    def _get_camera_location(self, camera_id: str) -> Dict:
        """Get camera location information."""
        # TODO: Implement camera location tracking
        return {'latitude': 0.0, 'longitude': 0.0}
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the latest processed frame for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Processed frame or None if not available
        """
        try:
            if camera_id in self.frame_queues and not self.frame_queues[camera_id].empty():
                return self.frame_queues[camera_id].get_nowait()
            return None
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {str(e)}")
            return None
    
    def start(self):
        """Start all camera streams and processing."""
        self.is_running = True
        logger.info("Camera manager started")
    
    def stop(self):
        """Stop all camera streams and processing."""
        self.is_running = False
        
        # Stop all cameras
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Camera manager stopped") 