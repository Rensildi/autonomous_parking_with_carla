import carla

# Connect to the CARLA server
client = carla.CLient('localhost', 2000)
client.set_timeout(5.0)

# Get the world (environment) in the simulation
world = client.get_world()
print(f"Connected to CARLA world: {world.get_map().name}")
