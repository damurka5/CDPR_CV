import numpy as np

def compute_cable_lengths(ee_pos, anchor_points, box_size, pulley_radius):
    """
    Calculate cable lengths considering pulley wrapping
    """
    # Implement your cable length calculation logic here
    results = {}
    
    # Example for one cable:
    A = ee_pos + np.array([-box_size[0], 0, box_size[2]])
    beta = np.arctan2(A[1] - anchor_points[0][1], A[0] - anchor_points[0][0])
    C_c = anchor_points[0] + np.array([pulley_radius*np.cos(beta), 
                                      pulley_radius*np.sin(beta), 0])
    
    L = np.linalg.norm(A - C_c)
    eps = np.arccos(pulley_radius / L)
    delta = np.arccos(np.sqrt((A[0]-C_c[0])**2 + (A[1]-C_c[1])**2) / L)
    gamma = eps - delta
    
    B = C_c + np.array([pulley_radius*np.cos(gamma)*np.cos(beta),
                       pulley_radius*np.cos(gamma)*np.sin(beta),
                       pulley_radius*np.sin(gamma)])
    
    cable_length = pulley_radius * (np.pi - gamma) + np.linalg.norm(A - B)
    
    return cable_length