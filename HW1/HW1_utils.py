# Utilities file for functions used in HW1 main code

import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(ax, cov, center, t):
    """ 
    Plot an ellipse based on a covariance matrix in matplotlib
    Inputs: covariance matrix, center
    Outputs: plotted ellipse
    """
    eig_vals, eig_vect = np.linalg.eig(cov)

    # Get smallest and largest eigenvectors/values
    max_ind = np.argmax(eig_vals) # USE ABSOLUTE VALUE?
    # print("\n Max eig value indice =", max_ind)
    max_EV = eig_vect[:,max_ind]
    max_eval = np.max(eig_vals)
    min_eval = np.min(eig_vals)

    # Obtain minimum eigenvectors
    if max_ind == 0:
        min_EV = eig_vect[:,1]
    else:
        min_EV = eig_vect[:,0]
    
    # Angle between x-axis and largest eigenvector
    angle = np.arctan2(max_EV[1], max_EV[0])
    if angle < 0: angle += 2*np.pi

    # Mean data coords
    center = center
    chi_sq = 1

    a = chi_sq * np.sqrt(max_eval)
    b = chi_sq * np.sqrt(min_eval)
    theta = np.linspace(0, 2*np.pi, 1000);
    pts_x = np.array(a * np.cos(theta))
    pts_y = np.array(b * np.sin(theta))
    pts_tot = np.vstack([pts_x, pts_y])
    pts_tot = pts_tot.T

    # Rotation matrix 
    R = [[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]]
    r_ellipse = pts_tot @ R

    plt.plot(r_ellipse[:,0] + center[0], r_ellipse[:,1] + center[1], label = f't={t}')

    return ax

def kalman_filter_pred(A, B, R, state_prev, post_prev):
    """ Simulates Kalman filter prediction steps: ALGORITHM LINES 2-3
    Assumptions: deltaT = 1, system starts at REST, acceleration N~(0,1)
    Input: A, B, R matrices, previous state and posterior
    Output: predicted states, covariance, and acceleration for current state
    """

    # Random sample acceleration
    u_t = np.random.normal(0, 1, size=(1,1))
    acc = u_t.item()

    # State prediction 
    state_new_p = A @ state_prev + B @ u_t

    # Posterior 
    post_new_p = A @ post_prev @ A.T + R

    return state_new_p, post_new_p, acc

def kalman_filter_prob2_2_GPS(A, B, C, Q, R, state_prev, post_prev, t):
    """ ONLY FOR PROBLEM 2.2: ASKS FOR A MEASUREMENT OF 10 at TIME T=5
    Simulates Kalman filter prediction and measurement update steps: ALGORITHM LINES 2-6
    Assumptions: deltaT = 1, system starts at REST, acceleration N~(0,1)
    Input: A, B, C, Q, R matrices, previous state and posterior, GPS sensor probability
    Output: updated state prediction, covariance, and acceleration for current state
    """

    # PREDICTION STEPS: LINES 2-3
    # Random sample for accel using normal dist
    u_t = np.random.normal(0, 1, size=(1,1))
    acc = u_t.item()
    
    # State prediction 
    state_new_p = A @ state_prev + B @ u_t

    # Posterior 
    post_new_p = A @ post_prev @ A.T + R

    # MEASURMENT UPDATE STEPS: LINES 4-6
    # Only happens at time t = 5
    if t == 5:
        print("\n State prediction before measurement at t=5 \n", state_new_p, "\n")
        # print("\n Posterior before measurement \n", post_new_p)
        # Sample TRUE POSITION - 10 in this case
        z_t = np.array([[10]])
        print("\n Sensor reading = ", z_t, "\n")
        
        # Kalman gain 
        K_t = post_new_p @ C.T @ np.linalg.inv(C @ post_new_p @ C.T + Q) 
        # print("\n Kalman gain \n",K_t)

        # State update
        # print("\n checkpoint1 = ", z_t - C @ state_new_p)
        # print("\n checkpoint2 = ", K_t@(z_t - C @ state_new_p))
        state_new = state_new_p + K_t @ (z_t - C @ state_new_p)
        print("\n New state \n", state_new)

        # Covariance update 
        post_new = (np.eye(2) - K_t @ C) @ post_new_p
        # print("\n New posterior \n", post_new)
    else:
        # Add state prediction and posterior 
        state_new = state_new_p
        post_new = post_new_p


    return state_new, post_new, acc

def kalman_filter_randGPS(A, B, C, Q, R, state_prev, post_prev, GPS_prob, true_position,i):
    """ Simulates Kalman filter prediction and measurement update steps: ALGORITHM LINES 2-6
    Assumptions: deltaT = 1, system starts at REST, acceleration N~(0,1)
    Input: A, B, C, Q, R matrices, previous state and posterior, GPS sensor probability
    Output: updated state prediction, covariance, and acceleration for current state
    """
    sigma_pos = 8
    # PREDICTION STEPS: LINES 2-3
    # Random sample for accel using normal dist
    u_t = np.random.normal(0, 1, size=(1,1))
    acc = u_t.item()
    
    # State prediction 
    state_new_p = A @ state_prev + B @ u_t

    # Posterior 
    post_new_p = A @ post_prev @ A.T + R

    # MEASURMENT UPDATE STEPS: LINES 4-6
    # First - if the GPS sensor 'works', get a measurement
    GPS_result = np.random.choice([False, True], p=[GPS_prob, 1-GPS_prob])
    # print(result)

    if GPS_result == True:
        # print("\n State prediction before measurement \n", state_new_p, "\n")
        # print("\n Posterior before measurement \n", post_new_p)
        # Sample TRUE POSITION
        pos = true_position[i]
        z_t = np.random.normal(pos, sigma_pos, size=(1,1))
        # print("\n Sensor reading = ", z_t, "\n")
        
        # Kalman gain 
        K_t = post_new_p @ C.T @ np.linalg.inv(C @ post_new_p @ C.T + Q) 
        # print("\n Kalman gain \n",K_t)

        # State update
        # print("\n checkpoint1 = ", z_t - C @ state_new_p)
        # print("\n checkpoint2 = ", K_t@(z_t - C @ state_new_p))
        state_new = state_new_p + K_t @ (z_t - C @ state_new_p)
        # print("\n New state \n", state_new)

        # Covariance update 
        post_new = (np.eye(2) - K_t @ C) @ post_new_p
        # print("\n New posterior \n", post_new)
    else:
        # Add state prediction and posterior 
        state_new = state_new_p
        post_new = post_new_p
        # error = np.abs(state_new[0] - true_position[i])


    return state_new, post_new, acc, GPS_result



# Simulate ground truth dynamics with wind acceleration
def simulate_ground_truth(t_max, mu_wind=0.0, sigma_wind=1.0):
    """
    Simulate the ground truth position and velocity over time with random wind acceleration.
    Inputs: Maximum time step, Mean wind acceleration (default = 0), st. dev of wind
    Output: Ground truth position and velocity arrays over time
    """
    # Initialize arrays to store true position and velocity
    true_position = np.zeros(t_max + 1)
    true_velocity = np.zeros(t_max + 1)

    for t in range(1, t_max + 1):
        # Sample random acceleration from wind
        acceleration = np.random.normal(mu_wind, sigma_wind)
        
        # Update velocity and position using the true dynamics
        true_velocity[t] = true_velocity[t-1] + acceleration  # Update velocity
        true_position[t] = true_position[t-1] + true_velocity[t]  # Update position

    return true_position, true_velocity

def kalman_filter_pred_prob3(A, B, R, state_prev, post_prev):
    """ Simulates Kalman filter prediction steps: ALGORITHM LINES 2-3
    Assumptions: deltaT = 1, system starts at REST, acceleration N~(0,1)
    Input: A, B, R matrices, previous state and posterior
    Output: predicted states, covariance, and acceleration for current state
    """

    # Random sample acceleration
    # u_t = np.random.normal(0, 1, size=(1,1))
    u_t = np.array([[1]])
    acc = u_t.item()

    # State prediction 
    state_new_p = A @ state_prev + B @ u_t

    # Posterior 
    post_new_p = A @ post_prev @ A.T + R

    return state_new_p, post_new_p, acc