# music to chill https://www.youtube.com/watch?v=LBDCQfju1cY
# carla_parking_system.py
import carla
import pygame
import numpy as np
import cv2
import random
import time
import math
import zmq  # For communication with YOLOv8 process
from collections import deque

# Constants
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
YOLO_COMM_PORT = 5555  # For ZeroMQ communication

class CarlaParkingSystem:
    def __init__(self):
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        
        # Setup ZeroMQ context for communicating with YOLOv8
        self.setup_yolo_communication()
        
        # Load world and setup vehicle/sensors
        self.world = self.client.load_world('Town04')
        time.sleep(5.0)
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.setup_pygame()
        self.setup_vehicle()
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle!")
            
        self.setup_sensors()
        
        # Initialize parking control attributes
        self.parking_state = "SEARCHING"
        self.target_parking_spot = None
        self.current_detections = []
        self.control_sequence = deque()
        
        # For visualization
        self.font = pygame.font.SysFont('Arial', 20)
        self.camera_surfaces = {}
        
    def setup_yolo_communication(self):
        """Setup ZeroMQ socket for communicating with YOLOv8 process"""
        self.yolo_context = zmq.Context()
        self.yolo_socket = self.yolo_context.socket(zmq.REQ)
        self.yolo_socket.connect(f"tcp://localhost:{YOLO_COMM_PORT}")
        self.yolo_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.yolo_socket.setsockopt(zmq.LINGER, 0)  # Don't linger on close
    def setup_pygame(self):
        """Initialize pygame display"""
        pygame.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Autonomous Parking System - CARLA")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.camera_surfaces = {}
        
    def setup_vehicle(self):
        """Spawn the ego vehicle with retry logic"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        
        for attempt in range(5):
            spawn_point = random.choice(self.spawn_points) if attempt > 0 else carla.Transform(
                carla.Location(x=-30, y=-132, z=0.5),
                carla.Rotation(yaw=180)
            )
            
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break
            time.sleep(1.0)
        
        if self.vehicle is None:
            return
            
        physics_control = self.vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        self.vehicle.apply_physics_control(physics_control)
        self.vehicle.set_autopilot(True)
        time.sleep(2.0)
        self.vehicle.set_autopilot(False)
        
    def setup_sensors(self):
        """Setup all required sensors with improved configurations"""
        blueprint_library = self.world.get_blueprint_library()
        
        camera_configs = {
            'front': {'transform': carla.Transform(carla.Location(x=2.5, z=1.5)), 'fov': 90},
            'rear': {'transform': carla.Transform(carla.Location(x=-2.5, z=1.5), carla.Rotation(yaw=180)), 'fov': 90},
            'left': {'transform': carla.Transform(carla.Location(y=-1.0, z=1.5), carla.Rotation(yaw=-90)), 'fov': 90},
            'right': {'transform': carla.Transform(carla.Location(y=1.0, z=1.5), carla.Rotation(yaw=90)), 'fov': 90},
            'bev_semantic': {'transform': carla.Transform(carla.Location(z=15), carla.Rotation(pitch=-90)), 
                            'type': 'sensor.camera.semantic_segmentation', 'fov': 90},
            'attention': {'transform': carla.Transform(carla.Location(x=1.5, z=1.7)), 'fov': 60}
        }

        self.cameras = {}
        for name, config in camera_configs.items():
            bp = blueprint_library.find(config.get('type', 'sensor.camera.rgb'))
            bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
            bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
            bp.set_attribute('fov', str(config['fov']))
            
            camera = self.world.spawn_actor(
                bp,
                config['transform'],
                attach_to=self.vehicle
            )
            camera.listen(lambda image, n=name: self.process_camera_image(image, n))
            self.cameras[name] = camera
    
    def process_camera_image(self, image, camera_name):
        """Process camera images and optionally send to YOLOv8"""
        if camera_name == 'front':  # Only process front camera with YOLOv8
            # Convert to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Drop alpha channel
            
            # Send to YOLOv8 process for detection
            self.send_to_yolo(array, camera_name)
            
        # For visualization
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # Convert to RGB
        
        if camera_name == 'attention':
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            array = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        self.camera_surfaces[camera_name] = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def send_to_yolo(self, image_array, camera_name):
        """Send image to YOLOv8 process and get detections"""
        try:
            # Serialize image and send
            self.yolo_socket.send_pyobj({
                'camera': camera_name,
                'image': image_array,
                'timestamp': time.time()
            })
            
            # Get response with detections (non-blocking with timeout)
            if self.yolo_socket.poll(timeout=100):  # 100ms timeout
                detections = self.yolo_socket.recv_pyobj()
                self.process_yolo_detections(detections)
                
        except zmq.ZMQError as e:
            print(f"YOLO communication error: {e}")
    
    def process_yolo_detections(self, detections):
        """Process parking spot detections from YOLOv8"""
        if not detections or 'parking_spots' not in detections:
            return
            
        # Transform detections to world coordinates
        vehicle_transform = self.vehicle.get_transform()
        camera_transform = self.cameras['front'].get_transform()
        
        self.current_detections = []
        for spot in detections['parking_spots']:
            # Convert from image coordinates to world coordinates
            # (This would use camera intrinsics/extrinsics in a real implementation)
            world_spot = self.image_to_world(spot, vehicle_transform, camera_transform)
            self.current_detections.append(world_spot)
        
        # Update parking state if searching
        if self.parking_state == "SEARCHING" and self.current_detections:
            self.select_best_parking_spot()
    
    def image_to_world(self, spot, vehicle_tf, camera_tf):
        """Convert image detection to world coordinates (simplified)"""
        # Simplified transformation - in a real system you'd use:
        # 1. Camera intrinsics to get 3D ray
        # 2. Camera->Vehicle transform
        # 3. Vehicle->World transform
        # 4. Ground plane intersection
        
        # For demo, we'll assume spots are on the ground plane to the right
        spot_distance = 8.0  # meters
        spot_angle = math.radians(vehicle_tf.rotation.yaw - 90)
        
        return {
            'center': carla.Location(
                x=vehicle_tf.location.x + math.cos(spot_angle) * spot_distance,
                y=vehicle_tf.location.y + math.sin(spot_angle) * spot_distance,
                z=vehicle_tf.location.z
            ),
            'length': 6.0,
            'width': 2.5,
            'angle': vehicle_tf.rotation.yaw - 90,
            'type': 'parallel',
            'confidence': spot.get('confidence', 0.9)
        }
    
    def select_best_parking_spot(self):
        """Select the best parking spot from available detections"""
        if not self.current_detections:
            return
            
        # Simple selection logic - choose closest high-confidence spot
        self.target_parking_spot = max(
            self.current_detections,
            key=lambda x: x['confidence']
        )
        
        self.parking_state = "APPROACHING"
        print(f"Selected parking spot at {self.target_parking_spot['center']}")
        
        # Plan parking path
        self.plan_parking_maneuver()
    
    def plan_parking_maneuver(self):
        """Plan appropriate parking maneuver based on spot type"""
        if not self.target_parking_spot:
            return
            
        if self.target_parking_spot['type'] == 'parallel':
            self.control_sequence = deque(self.parallel_parking_sequence())
        elif self.target_parking_spot['type'] == 'perpendicular':
            self.control_sequence = deque(self.perpendicular_parking_sequence())
        else:
            self.control_sequence = deque(self.angle_parking_sequence())
    
    def parallel_parking_sequence(self):
        """Generate control sequence for parallel parking"""
        return [
            {'throttle': 0.5, 'steer': 0.0, 'brake': 0.0, 'reverse': False, 'duration': 2.0},
            {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0, 'reverse': False, 'duration': 1.0},
            {'throttle': 0.3, 'steer': -0.5, 'brake': 0.0, 'reverse': True, 'duration': 2.0},
            {'throttle': 0.3, 'steer': 0.5, 'brake': 0.0, 'reverse': True, 'duration': 2.0},
            {'throttle': 0.2, 'steer': 0.0, 'brake': 0.0, 'reverse': True, 'duration': 1.0},
            {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0, 'reverse': False, 'duration': 1.0}
        ]
    
    # Add similar methods for other parking types...
    
    def update_parking_behavior(self):
        """Update vehicle control based on parking state"""
        if self.parking_state == "SEARCHING":
            return  # Waiting for YOLOv8 detections
            
        elif self.parking_state in ["APPROACHING", "PARKING"]:
            if self.control_sequence:
                current_control = self.control_sequence[0]
                
                control = carla.VehicleControl()
                control.throttle = current_control['throttle']
                control.steer = current_control['steer']
                control.brake = current_control['brake']
                control.reverse = current_control['reverse']
                self.vehicle.apply_control(control)
                
                current_control['duration'] -= 0.05
                if current_control['duration'] <= 0:
                    self.control_sequence.popleft()
                    
                    if not self.control_sequence:
                        self.parking_state = "PARKED"
                        print("Parking completed!")
    
    def render(self):
        """Render all displays with improved visualization"""
        self.display.fill((50, 50, 50))
        
        # Camera layout
        camera_positions = {
            'front': (0, 0),
            'right': (CAMERA_WIDTH, 0),
            'rear': (CAMERA_WIDTH * 2, 0),
            'left': (CAMERA_WIDTH * 3, 0),
            'bev_semantic': (0, CAMERA_HEIGHT),
            'attention': (CAMERA_WIDTH, CAMERA_HEIGHT)
        }
        
        for name, pos in camera_positions.items():
            if name in self.camera_surfaces:
                self.display.blit(self.camera_surfaces[name], pos)
                label = self.font.render(name.upper(), True, (255, 255, 255))
                self.display.blit(label, (pos[0] + 10, pos[1] + 10))
        
        # Draw parking status and detections
        self.draw_parking_info()
        pygame.display.flip()
    
    def draw_parking_info(self):
        """Draw parking status and detection information"""
        status_text = f"Parking Status: {self.parking_state}"
        if self.target_parking_spot:
            loc = self.target_parking_spot['center']
            status_text += f" | Spot: {self.target_parking_spot['type']} at ({loc.x:.1f}, {loc.y:.1f})"
        
        status_label = self.font.render(status_text, True, (255, 255, 255))
        self.display.blit(status_label, (10, WINDOW_HEIGHT - 30))
        
        # Draw detection count if available
        if hasattr(self, 'current_detections'):
            detections_text = f"Detections: {len(self.current_detections)} spots"
            detections_label = self.font.render(detections_text, True, (255, 255, 255))
            self.display.blit(detections_label, (10, WINDOW_HEIGHT - 60))
    
    def run(self):
        """Main simulation loop"""
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                
                self.update_parking_behavior()
                self.world.tick()
                self.render()
                self.clock.tick(20)
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        for camera in self.cameras.values():
            camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        pygame.quit()
        self.yolo_socket.close()
        self.yolo_context.term()

if __name__ == "__main__":
    try:
        parking_system = CarlaParkingSystem()
        parking_system.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()