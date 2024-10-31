# Utilities File 
# Functions required for particle filter HW2

import numpy as np 
import matplotlib.pyplot as plt
import cv2


def random_movement():
    """ Generates a random movement vector in the x and y direction, 
    such that dx^2 + dy^2 = 1.0
    Inputs: none
    Outputs: dx, dy
    """
    dx = np.random.uniform(-1, 1)
    posneg = np.random.choice([-1, 1])
    dy = (np.sqrt(1-dx**2))*posneg
    return dx,dy

def move_drone(pos,map, sigma_movement = 5):
    """ 
    Moves the drone a random distance dx, dy
    while checking to make sure it is in range of the map
    Inputs: Previous position, map
    OUtputs: new position 
    """
    dx,dy = random_movement()
    m = 100
    # print(dx)
    # print(dy)
    move_x = int(np.floor(dx*50)) + np.random.randn()*sigma_movement
    move_y = int(np.floor(dy*50)) + np.random.randn()*sigma_movement
    new_pos = pos + np.array([[move_x],[move_y]])

    while new_pos[0]+m/2 > (map.shape[1]) or new_pos[1]+m/2 > (map.shape[0]) or new_pos[0]-m/2 < (0) or new_pos[1]-m/2 < (0):
        print("Movement rejected, generating a new movement")
        dx,dy = random_movement()
        move_x = int(np.floor(dx*50)) + sigma_movement
        move_y = int(np.floor(dy*50)) + sigma_movement
        new_pos = pos + np.array([[move_x],[move_y]])
        # print(new_pos)

    return new_pos,dx,dy

def move_particle(pos,map, dx, dy, sigma_particle = 3):
    """ 
    Moves the drone a random distance dx, dy
    while checking to make sure it is in range of the map
    Inputs: Previous position, map
    OUtputs: new position 
    """
    # print(pos)
    move_x = int(np.floor(dx*50))
    # print(move_x)
    move_y = int(np.floor(dy*50))
    # print(move_y)
    new_pos = pos + np.array([[move_x],[move_y]])
    # print(new_pos)

    return new_pos

def im_comp(ref,particle,m):
    """ 
    Compares the view from a particle and compares 
    it to a reference image
    Inputs: Reference image, particle, pixels in each direction (m)
    Outputs: error associated with that particle
    Using ABSOLUTE ERROR APPROACH 
    """
    max_error = 10000000
    x = int(particle[0])
    y = int(particle[1])
    par_im = map[y:y+m,x:x+m]
    # Check if the extracted image has the required shape
    if par_im.shape != (100, 100, 3):
        # par_im = cv2.resize(par_im, (100, 100))
        return max_error  # Assign max error if shape is incorrect
        
    diff = cv2.absdiff(ref, par_im) # Absolute difference
    error = np.sum(diff) # Sum errors 
    
    return error

def low_variance_sampler(particles, weights):
    """
    Low variance resampling based on the particle weights.
    Parameters:
    particles (numpy.ndarray): Array of particle positions (N x 3)
    weights (numpy.ndarray): Normalized weights associated with each particle
    Returns:
    numpy.ndarray: Resampled array of particles
    FROM TEXTBOOK: Thrun, Burgard, Fox, page 86
    """
    N = len(particles)
    new_particles = np.zeros_like(particles)
    cumulative_sum = np.cumsum(weights)
    # print(cumulative_sum)
    
    # Start uniformly between 0 and 1/N
    start = np.random.uniform(0, 1 / N)
    index = 0  # Index of particles
    
    # Low variance saEmpling loop
    for i in range(N):
        u = start + i * (1 / N)
        while u > cumulative_sum[index]:
            index += 1
        new_particles[i] = particles[index]
        
    return new_particles

def GetScanRelation(BinsPerChannel, ref, particle, map):
    """ 
    Generates a 3D color histogram comparison of the reference image and particle image.
    
    Inputs: BinsPerChannel: Number of bins per channel for histogram calculation
        ref: Reference image
        particle: Particle position as an array containing (x, y)
    
    Output: Correlation score between the reference and particle image histogram
    """
    # Extract particle position
    x = int(particle[0])
    y = int(particle[1])
    
    # Define the size of the patch to extract
    m = 100  # Size of the patch (100x100)
    
    # Ensure the extracted area is within bounds
    if (y >= m // 2 and y <= map.shape[0] - m // 2) and (x >= m // 2 and x <= map.shape[1] - m // 2):
        # Extract the image patch from the map
        par_im = map[y - m // 2:y + m // 2, x - m // 2:x + m // 2]
        # print("PAR IMAGE", par_im)
        # print("\n REF \n", ref)
        # Calculate histograms
        histref = cv2.calcHist([ref], [0, 1, 2], None, [BinsPerChannel, BinsPerChannel, BinsPerChannel],
                                [0, 256, 0, 256, 0, 256])
        histp = cv2.calcHist([par_im], [0, 1, 2], None, [BinsPerChannel, BinsPerChannel, BinsPerChannel],
                              [0, 256, 0, 256, 0, 256])

        # Normalize histograms
        histref = cv2.normalize(histref, histref).flatten()
        histp = cv2.normalize(histp, histp).flatten()

        # Return the correlation between the two histograms
        return cv2.compareHist(histref, histp, cv2.HISTCMP_CORREL)
    else:
        # If the particle position is out of bounds, return a default value (e.g., 0 or max correlation)
        return 0

def abs_avg_dev(pos_arr, drone_pos):
    """ 
    Calculates the absolute average deviation of a pos array of particle 
    Inputs: Array of particle positions, drone position 
    Outputs: absolute average deviation 
    """
    absdev = 0
    N = len(pos_arr)
    for i in range(N):
        p = pos_arr[i,1:].reshape(2,1)
        # print(p)
        # print(drone_pos)
        x_dif = p[0] - drone_pos[0]
        y_dif = p[1] - drone_pos[1]
        dev = np.sqrt(x_dif**2 + y_dif**2)
        dev_w = 1/N * dev
        absdev += dev_w
        
    return absdev
        
        
