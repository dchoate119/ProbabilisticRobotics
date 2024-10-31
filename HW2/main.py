# Probabilistic Robotics HW2
# Daniel Choate 
# CS-141

import numpy as np 
import matplotlib.pyplot as plt
import cv2
from HW2_utils import *

# Upload image of choice 
map_orig = cv2.imread('BayMap.png')
# map_orig = cv2.imread('CityMap.png')
# map_orig = cv2.imread('MarioMap.png')
map_orig = cv2.cvtColor(map_orig, cv2.COLOR_BGR2RGB)

map = map_orig.copy()

# Constants

dFOV = 50  # Field of view of the drone
sigma_movement = 5  # Movement from wind

# Calculate the range in the x and y directions
range_x = ((map.shape[1] - dFOV) // 2) // 50
range_y = ((map.shape[0] - dFOV) // 2) // 50
print(f'\n The range in the x and y directions is ({range_x},{range_y}) units \n')

# Simulate a drone's starting position x and y
pos_x = np.random.randint(-range_x, range_x + 1)
pos_y = np.random.randint(-range_y, range_y + 1)

state_ini = np.array([[pos_x], [pos_y]])
print(f'\n The initial position vector x = {state_ini.flatten()} \n')

# Convert unit position to pixel coordinates
center_x = int((pos_x) * 50 + (map.shape[1] // 2))
center_y = int((range_y+1 - pos_y) * 50)
ini_ref_pos = np.array([[center_x],[center_y]])
print(ini_ref_pos)

# Draw a circle with radius 50 pixels (1 unit)
radius_pixels = 50  # 1 unit = 50 pixels
cv2.circle(map, (center_x, center_y), radius_pixels, (255, 0, 0), 2)  # Red circle

# Show the image
plt.figure(figsize=(15, 8))
plt.title("Original Drone Position")
plt.imshow(map)
plt.axis("off")
plt.show()


# Initialize parameters
m = 100
N = 100 #N*N particles, subject to change
iterations = 20
BinsPerChannel = 8
trials = 1 # Change based on test experiment

# Run the particle filter for each value of m
avg_deviations = {}  # Dictionary to store deviations for each m value
for m in [100]: #[50, 100, 150]: Uncomment to test all three trials
    deviations_per_trial = []  # Store deviations for each trial    
    for t in range(trials):
        
        # Initialize map
        map = map_orig.copy()
        hal = int(m/2)
        ini_pos = map[ini_ref_pos[1,0]-hal:ini_ref_pos[1,0]+hal, ini_ref_pos[0,0]-hal:ini_ref_pos[0,0]+hal] #, height_s:height_e]
    
        # Show the true position image
        plt.figure(figsize=(15, 8))
        plt.imshow(ini_pos)
        plt.title("Reference Image")
        plt.axis("off")
        plt.show()
        # Particle filter implementation
        # Re-initialize map
        dispmap = map.copy()
    
    
    
        # Generate a set P of N particles, randomly distributed across the map 
        particlePosArr = np.zeros([N**2, 3])  # particle number, xpos, ypos
        
        for particleCount in range(N**2):
            # Random x and y positions within the map, adjusted to stay within the FOV
            x_pos = np.random.uniform(0, map.shape[1])# - dFOV)
            y_pos = np.random.uniform(0, map.shape[0])# - dFOV)
            particlePosArr[particleCount, :] = [particleCount, x_pos, y_pos]
            
            # Draw the particle on the map image
            cv2.circle(dispmap, (int(x_pos), int(y_pos)), 2, (255, 0, 0), -1)  # Red circles for particles
    
    
        # Display the map with particles
        plt.imshow(dispmap)
        plt.title('Original Particles (no weight assigned)')
        plt.axis('off')  # Turn off the axis
        plt.show()
    
        devs_m = np.array([])
    
        # Loop through each iteration 
        ref_pos_pic = ini_pos
        ref_pos = ini_ref_pos
        # devs_50 = np.array([])
        # devs_100 = np.array([])
        # devs_150 = np.array([])
    
        for i in range(iterations):
            errors = []
            # First, create an image for each particle 
            for p in range(len(particlePosArr)):
                particle = particlePosArr[p,1:]
                error = GetScanRelation(BinsPerChannel, ref_pos_pic, particle, map)  # Use GetScanRelation for comparison
                error = error**2
                errors.append(error)
                
            weights = np.array(errors)
            weights_n = weights / np.sum(weights) # Normalize inverted weights 
            # print(weights_i_n)
            
        	map_with_particles = map.copy()
            scaled_weights = weights_n / weights_n.max()  # Scale to [0, 1] range for visibility
            
            # Use a low variance sampler
            particlePosArr = low_variance_sampler(particlePosArr, weights_n)
    
        
            
            for r, particle in enumerate(particlePosArr):
                x = int(particle[1])  # x position
                y = int(particle[2])  # y position
                size = int(5 + scaled_weights[r] * 5)  # Scale size based on weight, adjust as needed
                # cv2.circle(map_with_particles, (x, y), size, (255, 0, 0), 2)  # Draw filled circle in red
            # plt.imshow(map_with_particles)
            # plt.title(f'Resampled Particles, iteration {i+1}')
            # plt.show()
    
        
            map_with_particles = map.copy()
            # Now, move the particles and the reference image
            new_pos_drone,dx,dy = move_drone(ref_pos,map, sigma_movement = 5) # Move the drone, get dx, dy (DRONE WILL HAVE NOISE 
            ref_pos = new_pos_drone
        
            # Show the new position of the drone as a reference 
            mapdrone = map.copy()
            center_x = int(ref_pos[0])
            center_y = int(ref_pos[1])
            # Draw a circle with radius 50 pixels (1 unit)
            radius_pixels = 50  # 1 unit = 50 pixels
            cv2.circle(mapdrone, (center_x, center_y), radius_pixels, (255, 0, 0), 4)  # Red circle
            # Create new reference pic based on the moved drone 
            ref_pos_pic = map[center_y-hal:center_y+hal, center_x-hal:center_x+hal]
            # print(ref_pos_pic.shape)
    
        
            for k in range(len(particlePosArr)):
                p = particlePosArr[k,1:].reshape(2,1)
                new_pose_p = move_particle(p,map,dx,dy, sigma_particle=3)
                x = int(new_pose_p[0])  # x position
                y = int(new_pose_p[1])  # y position
                new_pose_p = new_pose_p.reshape(1,2)
                particlePosArr[k,1:] = new_pose_p
                size = int(5 + scaled_weights[k] * 5)
                cv2.circle(map_with_particles, (x, y), size, (255, 0, 0), 2)  # Draw filled circle in red
    
        
            cv2.circle(map_with_particles, (center_x, center_y), radius_pixels, (0, 255, 0), 4)  # GREEN circle
            plt.imshow(map_with_particles)
            plt.title(f'Moved Particles and true location, iteration {i+1}')
            plt.show(block=False)

            input("Press Enter to continue to the next iteration...")  # Pauses before each iteration
        
            # Calculate absolute average deviation
            absdev = abs_avg_dev(particlePosArr, ref_pos)
            devs_m = np.append(devs_m, absdev)
    
        deviations_per_trial.append(devs_m)
        # print(devs)

    avg_deviations[m] = np.mean(deviations_per_trial, axis=0)

# Print average deviations after each run
for m, avg_devs in avg_deviations.items():
    print(f"Average deviations for m={m}: {avg_devs}")



# Plot absolute average deviation vs. iterations for each m value (using averaged values)
plt.figure(figsize=(10, 6))

# Plot each averaged deviation set from the 'avg_deviations' dictionary
for m, avg_devs in avg_deviations.items():
    plt.plot(range(1, iterations + 1), avg_devs, marker='o', linestyle='-', label=f'm = {m}')

plt.xlabel("Iteration")
plt.ylabel(f'Absolute Average Deviation (Over {trials} Trials)')
plt.title("Average Absolute Deviation vs. Iterations")
plt.legend()
plt.grid(False)
plt.show()
