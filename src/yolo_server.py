# yolo_server.py
import os
import cv2
import zmq
import numpy as np
from ultralytics import YOLO

class YOLOParkingDetector:
    def __init__(self):
        # Load standard pretrained YOLOv8 model
        print("Loading YOLOv8 pretrained model...")
        self.model = YOLO('yolov8n.pt')  # Will auto-download if not found
        self.class_names = self.model.names
        print(f"Model loaded with classes: {self.class_names}")
        
        # Setup ZMQ communication
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("YOLOv8 server started on port 5555")
        
    def detect_parking_spots(self, image):
        """Detect cars and infer parking spots between them"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.model(image_rgb)
            
            # Get car detections (class 2 in COCO dataset)
            cars = []
            for result in results:
                for box in result.boxes:
                    if box.cls == 2:  # Class 2 is 'car' in COCO
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cars.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf[0])
                        })
            
            # Infer parking spots between cars (simple heuristic)
            parking_spots = self.infer_parking_spots(cars, image.shape)
            
            return {
                'parking_spots': parking_spots,
                'cars': cars  # Also return car detections for visualization
            }
        
        except Exception as e:
            print(f"Detection error: {e}")
            return {'error': str(e)}
    
    def infer_parking_spots(self, cars, image_shape):
        """Simple heuristic to find spaces between cars as parking spots"""
        if len(cars) < 2:
            return []
        
        # Sort cars by x1 coordinate (left to right)
        sorted_cars = sorted(cars, key=lambda x: x['bbox'][0])
        
        parking_spots = []
        img_height = image_shape[0]
        
        for i in range(len(sorted_cars)-1):
            car1 = sorted_cars[i]
            car2 = sorted_cars[i+1]
            
            # Calculate gap between cars
            gap = car2['bbox'][0] - car1['bbox'][2]
            
            # Only consider gaps larger than minimum parking spot width
            if gap > 50:  # pixels (adjust based on your needs)
                spot_bbox = [
                    car1['bbox'][2],  # x1 (right of left car)
                    0,                # y1 (top of image)
                    car2['bbox'][0],  # x2 (left of right car)
                    img_height        # y2 (bottom of image)
                ]
                
                parking_spots.append({
                    'bbox': spot_bbox,
                    'confidence': min(car1['confidence'], car2['confidence']),
                    'type': 'parallel'
                })
        
        return parking_spots
    
    def run(self):
        """Main detection loop"""
        print("YOLOv8 Parking Detector running...")
        while True:
            try:
                # Receive image from CARLA
                message = self.socket.recv_pyobj()
                print(f"Received image {message['camera']} at {message['timestamp']}")
                
                # Run detection
                detections = self.detect_parking_spots(message['image'])
                
                # Send back results
                self.socket.send_pyobj(detections)
                print(f"Found {len(detections.get('parking_spots', []))} parking spots")
                
            except Exception as e:
                print(f"Communication error: {e}")
                self.socket.send_pyobj({'error': str(e)})

if __name__ == "__main__":
    try:
        detector = YOLOParkingDetector()
        detector.run()
    except Exception as e:
        print(f"Failed to start YOLOv8 server: {e}")