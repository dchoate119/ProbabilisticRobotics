# Utilities file for functions used in HW1 main code

def plot_ellipse(cov, center, t):
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
    
    # Major and minor axes
    maj_len = 2*np.sqrt(lamdax)
    min_len = 2*np.sqrt(lamday)
    
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

    plt.plot(r_ellipse[:,0] + center[0], r_ellipse[:,1] + center[1], label = t)

    return ax