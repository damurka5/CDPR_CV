import numpy as np

# Parameters
radius = 0.2      # Radius of the circle (meters)
total_time = 10   # Total time for the trajectory (seconds)
sampling_freq = 3  # Sampling frequency (Hz)
center = [0, 0, 0.2]  # Center of the circle (X, Y, Z) in meters
num_points = total_time * sampling_freq  # Total data points

# Generate time, angle, and circular path
time = np.linspace(0, total_time, num_points)
theta = np.linspace(0, 2 * np.pi, num_points)  # Angle for full circle

# Circular trajectory equations (XY-plane, constant Z)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
z = np.full_like(x, center[2])  # Constant height

# Save to .txt file
filename = "cv/ee_circular_trajectory.txt"
with open(filename, 'w') as f:
    f.write("# Circular Trajectory Data for CDPR End-Effector\n")
    f.write("# Columns: Time(s), X(m), Y(m), Z(m)\n")
    for t, xi, yi, zi in zip(time, x, y, z):
        f.write(f"{xi:.4f} {yi:.4f} {zi:.4f}\n")

print(f"Trajectory saved to {filename}")