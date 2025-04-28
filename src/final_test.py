import carla
import pygame
import numpy as np
import cv2
import random
import time
import math
from collections import deque

# Constants
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300

class CarlaParkingSystem:
    def __init__(self):
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)  # Increased timeout
        
        # Load Town04
        self.world = self.client.load_world('Town04')
        time.sleep(5.0)  # Increased wait time for world to load
        
        # Get spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()
        
        # Setup pygame for display
        pygame.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Autonomous Parking System - CARLA")
        self.clock = pygame.time.Clock()
        
        # Setup vehicle and sensors
        self.vehicle = None
        self.setup_vehicle()
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts!")
            
        self.setup_sensors()
        
        # Parking control variables
        self.parking_state = "SEARCHING"
        self.target_parking_spot = None
        self.parking_path = []
        self.control_sequence = deque()
        
        # For visualization
        self.font = pygame.font.SysFont('Arial', 20)
        self.camera_surfaces = {}
        
    def setup_vehicle(self):
        """Spawn the ego vehicle with retry logic"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        
        # Try multiple spawn points if needed
        for attempt in range(5):  # Try up to 5 times
            spawn_point = random.choice(self.spawn_points) if attempt > 0 else carla.Transform(
                carla.Location(x=-30, y=-132, z=0.5),
                carla.Rotation(yaw=180)
            )
            
            print(f"Attempt {attempt + 1}: Trying to spawn at {spawn_point.location}")
            
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                print(f"Successfully spawned vehicle at {spawn_point.location}")
                break
            time.sleep(1.0)
        
        if self.vehicle is None:
            return
            
        # Set vehicle physics for better parking behavior
        physics_control = self.vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        self.vehicle.apply_physics_control(physics_control)
        
        # Set autopilot temporarily to avoid collisions during setup
        self.vehicle.set_autopilot(True)
        time.sleep(2.0)
        self.vehicle.set_autopilot(False)
        
    def setup_sensors(self):
        """Setup all required sensors for parking"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Camera configurations
        camera_configs = {
            'front': {'transform': carla.Transform(carla.Location(x=2.5, z=1.5)), 'fov': 90},
            'rear': {'transform': carla.Transform(carla.Location(x=-2.5, z=1.5), carla.Rotation(yaw=180)), 'fov': 90},
            'left': {'transform': carla.Transform(carla.Location(y=-1.0, z=1.5), carla.Rotation(yaw=-90)), 'fov': 90},
            'right': {'transform': carla.Transform(carla.Location(y=1.0, z=1.5), carla.Rotation(yaw=90)), 'fov': 90},
            'bev_semantic': {'transform': carla.Transform(carla.Location(z=15), carla.Rotation(pitch=-90)), 'type': 'sensor.camera.semantic_segmentation', 'fov': 90},
            'attention': {'transform': carla.Transform(carla.Location(x=1.5, z=1.7)), 'fov': 60}
        }

        
        self.cameras = {}
        
        for name, config in camera_configs.items():
            # Get the appropriate blueprint
            if 'type' in config:
                bp = blueprint_library.find(config['type'])
            else:
                bp = blueprint_library.find('sensor.camera.rgb')
            
            # Set camera attributes
            bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
            bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
            bp.set_attribute('fov', str(config['fov']))
            
            # Create transform
            transform = config['transform']
            if 'rotation' in config:
                transform.rotation = config['rotation']
            
            # Spawn and attach camera
            camera = self.world.spawn_actor(
                bp,
                transform,
                attach_to=self.vehicle
            )
            
            # Add callback for each camera
            camera.listen(lambda image, n=name: self.process_camera_image(image, n))
            self.cameras[name] = camera
    
    def process_camera_image(self, image, camera_name):
        """Process raw camera data into pygame surfaces"""
        if 'semantic' in camera_name:
            # Convert semantic segmentation to colorful representation
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        else:
            # Convert regular RGB image
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Drop alpha channel
        array = array[:, :, ::-1]  # Convert BGR to RGB
        
        # Apply some processing for attention view
        if camera_name == 'attention':
            # Simple edge detection for demonstration
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            array = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Create pygame surface
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.camera_surfaces[camera_name] = surface
    
    def detect_parking_spots(self):
        """Simple parking spot detection (would be replaced with actual CV algorithm)"""
        # In a real implementation, this would use camera/LiDAR data
        # For demo purposes, we'll just return a fake spot in front of the vehicle
        
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # Create a parking spot to the right side of the vehicle
        spot_angle = math.radians(vehicle_rotation - 90)
        spot_distance = 8.0
        spot_length = 6.0
        spot_width = 2.5
        
        spot_center = carla.Location(
            x=vehicle_location.x + math.cos(spot_angle) * spot_distance,
            y=vehicle_location.y + math.sin(spot_angle) * spot_distance,
            z=vehicle_location.z
        )
        
        return [{
            'center': spot_center,
            'length': spot_length,
            'width': spot_width,
            'angle': vehicle_rotation - 90,
            'type': 'parallel'
        }]
    
    def plan_parking_path(self, parking_spot):
        """Generate a simple parallel parking path"""
        # Get vehicle dimensions
        bb = self.vehicle.bounding_box
        vehicle_length = bb.extent.x * 2
        vehicle_width = bb.extent.y * 2
        
        # Get vehicle current state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        
        # Calculate approach position
        approach_distance = 5.0  # meters in front of parking spot
        approach_angle = math.radians(parking_spot['angle'] + 180)
        
        approach_point = carla.Location(
            x=parking_spot['center'].x + math.cos(approach_angle) * approach_distance,
            y=parking_spot['center'].y + math.sin(approach_angle) * approach_distance,
            z=parking_spot['center'].z
        )
        
        # Generate control sequence for parallel parking
        controls = []
        
        # 1. Approach the parking spot
        controls.append({
            'throttle': 0.5,
            'steer': 0.0,
            'brake': 0.0,
            'reverse': False,
            'duration': 2.0  # seconds
        })
        
        # 2. Stop before parking
        controls.append({
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 1.0,
            'reverse': False,
            'duration': 1.0
        })
        
        # 3. Reverse into the spot (first part)
        controls.append({
            'throttle': 0.3,
            'steer': -0.5,  # Turn wheels left
            'brake': 0.0,
            'reverse': True,
            'duration': 2.0
        })
        
        # 4. Straighten out
        controls.append({
            'throttle': 0.3,
            'steer': 0.5,  # Turn wheels right
            'brake': 0.0,
            'reverse': True,
            'duration': 2.0
        })
        
        # 5. Final adjustment
        controls.append({
            'throttle': 0.2,
            'steer': 0.0,
            'brake': 0.0,
            'reverse': True,
            'duration': 1.0
        })
        
        # 6. Stop
        controls.append({
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 1.0,
            'reverse': False,
            'duration': 1.0
        })
        
        return controls
    
    def update_parking_behavior(self):
        """Update the vehicle's parking behavior based on current state"""
        if self.parking_state == "SEARCHING":
            # Detect parking spots
            parking_spots = self.detect_parking_spots()
            
            if parking_spots:
                self.target_parking_spot = parking_spots[0]
                self.parking_state = "APPROACHING"
                print("Found parking spot! Starting approach...")
                
                # Plan the parking path
                self.control_sequence = deque(self.plan_parking_path(self.target_parking_spot))
        
        elif self.parking_state == "APPROACHING" or self.parking_state == "PARKING":
            if self.control_sequence:
                current_control = self.control_sequence[0]
                
                # Apply the control
                control = carla.VehicleControl()
                control.throttle = current_control['throttle']
                control.steer = current_control['steer']
                control.brake = current_control['brake']
                control.reverse = current_control['reverse']
                self.vehicle.apply_control(control)
                
                # Update duration
                current_control['duration'] -= 0.05  # assuming 20Hz update rate
                
                # Check if this control step is complete
                if current_control['duration'] <= 0:
                    self.control_sequence.popleft()
                    
                    if not self.control_sequence:
                        self.parking_state = "PARKED"
                        print("Successfully parked!")
            else:
                self.parking_state = "PARKED"
    
    def render(self):
        """Render all camera views and UI"""
        # Fill background
        self.display.fill((50, 50, 50))
        
        # Camera positions in the window
        camera_positions = {
            'front': (0, 0),
            'right': (CAMERA_WIDTH, 0),
            'rear': (CAMERA_WIDTH * 2, 0),
            'left': (CAMERA_WIDTH * 3, 0),
            'bev_semantic': (0, CAMERA_HEIGHT),
            'attention': (CAMERA_WIDTH, CAMERA_HEIGHT)
        }
        
        # Draw camera views
        for name, pos in camera_positions.items():
            if name in self.camera_surfaces:
                self.display.blit(self.camera_surfaces[name], pos)
                # Draw label
                label = self.font.render(name.upper(), True, (255, 255, 255))
                self.display.blit(label, (pos[0] + 10, pos[1] + 10))
        
        # Draw parking status
        status_text = f"Parking Status: {self.parking_state}"
        if self.target_parking_spot:
            status_text += f" | Spot: {self.target_parking_spot['type']}"
        
        status_label = self.font.render(status_text, True, (255, 255, 255))
        self.display.blit(status_label, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop"""
        try:
            while True:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                
                # Update parking behavior
                self.update_parking_behavior()
                
                # Update world
                self.world.tick()
                
                # Render everything
                self.render()
                self.clock.tick(20)  # 20 FPS
                
        finally:
            # Cleanup
            for camera in self.cameras.values():
                camera.destroy()
            self.vehicle.destroy()
            pygame.quit()

if __name__ == "__main__":
    try:
        parking_system = CarlaParkingSystem()
        parking_system.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()