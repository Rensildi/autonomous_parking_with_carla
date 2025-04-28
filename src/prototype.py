import carla
import random
import time

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load Town03 (with parking lots)
world = client.load_world('Town03')
time.sleep(2.0)

blueprint_library = world.get_blueprint_library()

# Get a random vehicle blueprint
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

# Choose spawn point
spawn_points = world.get_map().get_spawn_points()

spawn_point = random.choice(spawn_points)

# Or manually pick near parking
spawn_point.location.x = 60   # adjust x
spawn_point.location.y = -130 # adjust y
spawn_point.location.z = 0.5
spawn_point.rotation.yaw = 90

# Spawn vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print('Spawned vehicle:', vehicle.type_id)

# Attach a camera sensor to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')

# Set camera attributes (resolution etc.)
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

# Camera relative position to the vehicle
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  
# (In front of the hood, 2.4m high)

# Spawn camera
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Make camera listen to frames (otherwise it does nothing)
camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

# (optional) Change spectator to match car view
spectator = world.get_spectator()
spectator.set_transform(camera.get_transform())

print('Camera attached to vehicle.')

# Keep alive
try:
    while True:
        world.tick()
        time.sleep(0.05)
except KeyboardInterrupt:
    print('Destroyed actors.')
finally:
    camera.destroy()
    vehicle.destroy()
