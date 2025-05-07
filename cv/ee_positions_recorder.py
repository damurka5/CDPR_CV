import cv2
import cv2.aruco as aruco
import numpy as np
from transforms3d.euler import mat2euler
import csv
import time
from datetime import datetime

# Marker configuration
BASE_MARKER_ID = 2       # Reference marker (0,0,0 position)
EE_MARKER_ID = 3         # End effector marker
MARKER_LENGTH = 0.05     # 5cm in meters (same for both markers)

# Load camera calibration
calibration_data = np.load('cv/camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Initialize ArUco detector
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Recording variables
is_recording = False
recorded_data = []
recording_start_time = 0

# Create rotation matrix to align Z-axis upward (Y becomes Z, Z becomes -Y)
R_z_up = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

print("Robot End Effector Tracking with Z-axis Upward")
print("Base Marker ID:", BASE_MARKER_ID)
print("End Effector Marker ID:", EE_MARKER_ID)
print("Commands:")
print("  [r] - Start/Stop recording path")
print("  [q] - Quit program")

def save_to_csv(filename, data):
    """Save recorded data to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        writer.writerows(data)
    print(f"\nSaved {len(data)} data points to {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    current_time = time.time()
    
    # Detect all markers
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        # Initialize transformation matrices
        base_rvec = base_tvec = None
        ee_rvec = ee_tvec = None
        
        # Process each detected marker
        for i in range(len(ids)):
            marker_id = ids[i][0]
            
            # Estimate pose for each marker
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                [corners[i]], MARKER_LENGTH, camera_matrix, dist_coeffs
            )
            
            if marker_id == BASE_MARKER_ID:
                base_rvec = rvec[0][0]
                base_tvec = tvec[0][0]
                # Draw base marker with blue outline
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH/2)
                frame = aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[BASE_MARKER_ID]]), borderColor=(255, 0, 0))
            
            elif marker_id == EE_MARKER_ID:
                ee_rvec = rvec[0][0]
                ee_tvec = tvec[0][0]
                # Draw end effector marker with green outline
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH/2)
                frame = aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[EE_MARKER_ID]]), borderColor=(0, 255, 0))
        
        # Calculate relative position if both markers are detected
        if base_rvec is not None and ee_rvec is not None:
            # Convert rotation vectors to rotation matrices
            R_base, _ = cv2.Rodrigues(base_rvec)
            R_ee, _ = cv2.Rodrigues(ee_rvec)
            
            # Calculate relative transformation
            R_rel = np.dot(R_base.T, R_ee)  # Relative rotation
            t_rel = np.dot(R_base.T, (ee_tvec - base_tvec))  # Relative translation
            
            # Apply Z-up orientation transformation
            R_rel_zup = np.dot(R_rel, R_z_up)
            
            # Convert rotation matrix to Euler angles (degrees)
            euler_angles = np.degrees(mat2euler(R_rel_zup, 'sxyz'))  # XYZ convention
            
            # Reorder position components for Z-up (X remains X, Y becomes Z, Z becomes -Y)
            pos_zup = np.array([t_rel[0], -t_rel[2], t_rel[1]])
            
            # Display relative position (Z-up)
            cv2.putText(frame, f"Relative Position (Z-up):", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"X: {pos_zup[0]:.3f}m", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Y: {pos_zup[1]:.3f}m", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Z: {pos_zup[2]:.3f}m", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display relative orientation
            cv2.putText(frame, f"Relative Orientation:", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {euler_angles[0]:.1f}deg", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {euler_angles[1]:.1f}deg", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Yaw: {euler_angles[2]:.1f}deg", (10, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Record data if in recording mode
            if is_recording:
                elapsed_time = current_time - recording_start_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                recorded_data.append([
                    timestamp,
                    pos_zup[0], pos_zup[1], pos_zup[2],
                    euler_angles[0], euler_angles[1], euler_angles[2]
                ])
                # Visual feedback for recording
                cv2.circle(frame, (frame.shape[1] - 20, 20), 10, (0, 0, 255), -1)
    
    # Show recording status
    status = "Recording..." if is_recording else "Ready"
    cv2.putText(frame, f"Status: {status}", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show detection status
    if ids is None or BASE_MARKER_ID not in ids or EE_MARKER_ID not in ids:
        status = "Searching for markers..."
        if ids is not None:
            detected_ids = [id[0] for id in ids]
            status = f"Detected: {detected_ids} (need {BASE_MARKER_ID} & {EE_MARKER_ID})"
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Robot End Effector Tracking (Z-up)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Start/stop recording
        if is_recording:
            is_recording = False
            if recorded_data:
                filename = f"cv/ee_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                save_to_csv(filename, recorded_data)
                recorded_data = []
        else:
            is_recording = True
            recording_start_time = current_time
            print("\nStarted recording path...")
    elif key == ord('q'):  # Quit program
        if is_recording:
            is_recording = False
            if recorded_data:
                filename = f"cv/ee_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                save_to_csv(filename, recorded_data)
        break

cap.release()
cv2.destroyAllWindows()