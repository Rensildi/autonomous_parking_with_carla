import carla
import random
import time
import pygame
import sys
import numpy as np

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load Town04
world = client.load_world('Town04')

available_maps = client.get_available_maps()
for map_name in available_maps:
    print(map_name)

time.sleep(2.0)

blueprint_library = world.get_blueprint_library()

# Force spawn three Lincoln MKZ vehicles
vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]

# Get all spawn points in the map
spawn_points = world.get_map().get_spawn_points()

# Select three distinct spawn points
spawn_point_1 = carla.Transform(
    carla.Location(x=60, y=-100, z=0.5),
    carla.Rotation(yaw=90)
)
spawn_point_2 = carla.Transform(
    carla.Location(x=60, y=-105, z=0.5),
    carla.Rotation(yaw=90)
)
spawn_point_3 = carla.Transform(
    carla.Location(x=60, y=-110, z=0.5),
    carla.Rotation(yaw=90)
)

# Spawn three vehicles
vehicle_1 = world.try_spawn_actor(vehicle_bp, spawn_point_1)
vehicle_2 = world.try_spawn_actor(vehicle_bp, spawn_point_2)
vehicle_3 = world.try_spawn_actor(vehicle_bp, spawn_point_3)

if vehicle_1 is None or vehicle_2 is None or vehicle_3 is None:
    print("Failed to spawn one or more vehicles!")
    sys.exit(1)

print('Spawned vehicle 1 at:', spawn_point_1.location)
print('Spawned vehicle 2 at:', spawn_point_2.location)
print('Spawned vehicle 3 at:', spawn_point_3.location)

# Camera setup
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-10))

# Initialize pygame
pygame.init()
display = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Manual Control - View from Vehicle")
clock = pygame.time.Clock()

# Control objects
control_1 = carla.VehicleControl()
control_2 = carla.VehicleControl()
control_3 = carla.VehicleControl()

# Camera image placeholder
image_surface = None

# Camera callback
def process_image(image):
    global image_surface
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Drop alpha channel
    array = array[:, :, ::-1]  # Convert BGR -> RGB
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# Vehicle control state
active_vehicle = 1
active_camera = None
current_gear = 'f'  # 'f' for forward (default), 'r' for reverse

# Initialize camera for vehicle 1
active_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_1)
active_camera.listen(process_image)

def switch_camera(new_vehicle):
    """ Switch camera to the specified vehicle """
    global active_camera, active_vehicle
    active_vehicle = {vehicle_1: 1, vehicle_2: 2, vehicle_3: 3}[new_vehicle]
    
    if active_camera is not None:
        active_camera.stop()
        active_camera.destroy()
    
    active_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=new_vehicle)
    active_camera.listen(process_image)

def process_keys():
    global current_gear
    
    keys = pygame.key.get_pressed()
    
    # Initialize all controls
    for control in [control_1, control_2, control_3]:
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        control.reverse = (current_gear == 'r')  # Set reverse based on current gear
    
    # Apply controls based on active vehicle
    if active_vehicle == 1:
        if keys[pygame.K_w]:  # Throttle in current gear direction
            control_1.throttle = 0.6
        if keys[pygame.K_s]:  # Brake
            control_1.brake = 0.8
        if keys[pygame.K_a]:
            control_1.steer = -0.3
        if keys[pygame.K_d]:
            control_1.steer = 0.3
        vehicle_1.apply_control(control_1)
    elif active_vehicle == 2:
        if keys[pygame.K_w]:
            control_2.throttle = 0.6
        if keys[pygame.K_s]:
            control_2.brake = 0.8
        if keys[pygame.K_a]:
            control_2.steer = -0.3
        if keys[pygame.K_d]:
            control_2.steer = 0.3
        vehicle_2.apply_control(control_2)
    elif active_vehicle == 3:
        if keys[pygame.K_w]:
            control_3.throttle = 0.6
        if keys[pygame.K_s]:
            control_3.brake = 0.8
        if keys[pygame.K_a]:
            control_3.steer = -0.3
        if keys[pygame.K_d]:
            control_3.steer = 0.3
        vehicle_3.apply_control(control_3)

# Main game loop
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    switch_camera(vehicle_1)
                elif event.key == pygame.K_2:
                    switch_camera(vehicle_2)
                elif event.key == pygame.K_3:
                    switch_camera(vehicle_3)
                elif event.key == pygame.K_r:
                    current_gear = 'r'
                    print("Gear set to REVERSE")
                elif event.key == pygame.K_f:
                    current_gear = 'f'
                    print("Gear set to FORWARD")

        process_keys()

        if image_surface:
            display.blit(image_surface, (0, 0))

        pygame.display.flip()
        world.tick()
        clock.tick(60)

except KeyboardInterrupt:
    print('Exiting and destroying actors.')
finally:
    if active_camera is not None:
        active_camera.stop()
        active_camera.destroy()
    vehicle_1.destroy()
    vehicle_2.destroy()
    vehicle_3.destroy()
    pygame.quit()