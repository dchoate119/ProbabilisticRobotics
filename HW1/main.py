from HW1_utils import *
import numpy as np
import matplotlib.pyplot as plt

# Question 1.3: Implement state prediction step of Kalman filter

# Assume at t=0, pos, vel, and accel = 0
# Calculate state disrtibution for times t = 1,2,...,5

# Time settings
dt = 1 # Assume delta t is 1 second
t_total = 5 # Total time in seconds
t_steps = int(t_total/dt) # Time steps 

muh_a = 0 # avg accel
sigma_a = 1 

# State transition, linear matrices A and B
A = np.array([[1, dt], [0, 1]])
B = np.array([[.5*dt**2], [dt]])
R = sigma_a * np.array([[.25, .5],[.5, 1]])

# Initialize resting state 
x_0 = np.array([[0],[0]])
cov_0 = np.array([[0, 0],[0, 0]])

# Create a state vector for number of time steps t
states = [x_0]
posts = [cov_0]
accs = [0]

# Create a plot to display axes
plt.figure()
ax = plt.gca()

# For times t=1,2, ..., 5
for i in range(1,t_steps+1):
    # Previous states
    state_prev = states[-1]
    post_prev = posts[-1]
    
    # Generate state prediction and distribution using Kalman algorithm 
    state_new_p, post_new_p, acc = kalman_filter_pred(A, B, R, state_prev, post_prev)

    # Plot an ellipse representing covariance 
    ax = plot_ellipse(ax, post_new_p, state_new_p, i)
    
    # Add state prediction, posterior, and accel
    states.append(state_new_p)
    posts.append(post_new_p)
    accs.append(acc)

states = np.hstack(states)
posts = np.hstack(posts)

print("\n States from t=0 to t=5 \n",states)
print("\n Posterior over x and x_dot \n",posts)
print("\n Accelerations t=0 to t=5 \n",accs)

plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Problem 1.3')

plt.legend(loc="upper left")
plt.show()




# Question 2-2: 
# Implement measurement update 

# Time settings
dt = 1 # Assume delta t is 1 second
t_total = 5 # Total time in seconds
t_steps = int(t_total/dt) # Time steps 

# Initialize resting state 
x_0 = np.array([[0],[0]])
cov_0 = ([0, 0],[0, 0])

# Create a state vector for number of time steps t
states = [x_0]
posts = [cov_0]
accs = [0]
GPS_results = []

muh_a = 0 # avg accel
sigma_a = 1 
sigma_pos = 8

# State transition 
# Linear matrices A and B
A = np.array([[1, dt], [0, 1]])
B = np.array([[.5*dt**2], [dt]])

# Implement C and Q matrices 
C = np.array([[1, 0]])
Q = np.array([[8]])

R = sigma_a * np.array([[.25, .5],[.5, 1]])
# print(R)

# Create plot
plt.figure()
ax = plt.gca()

for i in range(1, t_steps+1):

    # Initialize previous state
    state_prev = states[-1]
    post_prev = posts[-1]

    state_new, post_new, acc = kalman_filter_prob2_2_GPS(A, B, C, Q, R, state_prev, post_prev, i)
    
    # Add state prediction, posterior, accel, and GPS result
    states.append(state_new)
    posts.append(post_new)
    accs.append(acc)

    # Create a covariance ellipse
    ax = plot_ellipse(ax, post_new, state_new, i)


states = np.hstack(states)
posts = np.hstack(posts)

print("\n States from t=0 to t=5 \n",states)
print("\n Posterior over x and x_dot \n",posts)
print("\n Accelerations t=0 to t=5 \n",accs)

plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Problem 2.2')

plt.legend(loc="upper left")
plt.show()





# Question 2-3

# Simulate GROUND TRUTH for scenario (20 time steps) 
t_max = 20
true_position, true_velocity = simulate_ground_truth(t_max)

print(true_position)

# Implement measurement update 

# GPS SENSOR PROBABILITY 
sensor_probs = [0.1, 0.5, 0.9]

# Time settings
dt = 1 # Assume delta t is 1 second
t_total = 20 # Total time in seconds
t_steps = int(t_total/dt) # Time steps 

# Initialize resting state 
x_0 = np.array([[0],[0]])
cov_0 = ([0, 0],[0, 0])


muh_a = 0 # avg accel
sigma_a = 1 
sigma_pos = 8

# State transition 
# Linear matrices A and B
A = np.array([[1, dt], [0, 1]])
B = np.array([[.5*dt**2], [dt]])

# Implement C and Q matrices 
C = np.array([[1, 0]])
Q = np.array([[8]])

R = sigma_a * np.array([[.25, .5],[.5, 1]])
# print(R)


# Run N simulations for each scenario 
N = 100
results = {}

for sensor_prob in sensor_probs:
    
    errors = []
    
    for j in range(N):
        
        # Create a state vector for number of time steps t
        states = [x_0]
        posts = [cov_0]
        accs = [0]
        GPS_results = []
    
        for i in range(1, t_steps+1):
            # Initialize previous state
            state_prev = states[-1]
            post_prev = posts[-1]
        
            state_new, post_new, acc, GPS_result = kalman_filter_randGPS(
                A, B, C, Q, R, state_prev, post_prev, sensor_prob, true_position, i)
            
            # Add state prediction, posterior, accel, and GPS result
            states.append(state_new)
            posts.append(post_new)
            accs.append(acc)
            GPS_results.append(GPS_result)
        
            # # Create a covariance ellipse
            # ax = plot_ellipse(post_new, state_new, i)
        
        states = np.hstack(states)
        posts = np.hstack(posts)
        error = np.abs(states[0,20] - true_position[20])
        errors.append(error)

    results[sensor_prob] = np.mean(errors)
    print(f"Mean of the errors for sensor_prob {sensor_prob} = {results[sensor_prob]}")
        
    states = np.hstack(states)
    posts = np.hstack(posts)


# Extract sensor probabilities and corresponding average errors
x = sensor_probs
y = [results[sensor_prob] for sensor_prob in sensor_probs]

# Create bar graph
plt.figure(figsize=(8, 6))
plt.bar(x, y,width = 0.3)#, color=['blue', 'green', 'red'])
plt.xlabel('Sensor Probability')
plt.ylabel('Average Error')
plt.title('Average Error vs Sensor Probability')
plt.xticks(x)  # Set x-axis ticks to the sensor probabilities

# Display the plot
plt.show()



# # Question 3.1: Issuing Motor commands 
# Same as part 1, but B matrix CHANGES 
# Initial position and velocity become 5 and 1

# Assume at t=0, pos, vel, and accel = 0
# Calculate state disrtibution for times t = 1,2,...,5

# Time settings
dt = 1 # Assume delta t is 1 second
t_total = 1 # Total time in seconds
t_steps = int(t_total/dt) # Time steps 

muh_a = 0 # avg accel
sigma_a = 1 

# State transition, linear matrices A and B
A = np.array([[1, dt], [0, 1]])
# NEW B MATRIX 
B = np.array([[0],[1]])
R = sigma_a * np.array([[.25, .5],[.5, 1]])

# Initialize state with position and 
x_0 = np.array([[5],[1]])
cov_0 = np.array([[0, 0],[0, 0]])

# Create a state vector for number of time steps t
states = [x_0]
posts = [cov_0]
accs = [0]

# Create a plot to display axes
# plt.figure()
# ax = plt.gca()

# For times t=1,2, ..., 5
for i in range(1,t_steps+1):
    # Previous states
    state_prev = states[-1]
    # print(state_prev)
    post_prev = posts[-1]
    
    # Generate state prediction and distribution using Kalman algorithm 
    state_new_p, post_new_p, acc = kalman_filter_pred_prob3(A, B, R, state_prev, post_prev)
    
    # Add state prediction, posterior, and accel
    states.append(state_new_p)
    posts.append(post_new_p)
    accs.append(acc)

states = np.hstack(states)
posts = np.hstack(posts)

print("\n States from t=0 to t=1 \n",states)
print("\n Posterior over x and x_dot \n",posts)
