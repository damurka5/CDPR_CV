import cv2
import cv2.aruco as aruco
import numpy as np
from transforms3d.euler import mat2euler

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
cap = cv2.VideoCapture("cv/videos/webcam_20250508_143816.mp4")
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Tracking end effector relative to base marker")
print("Base Marker ID:", BASE_MARKER_ID)
print("End Effector Marker ID:", EE_MARKER_ID)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
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
            
            # Convert rotation matrix to Euler angles (degrees)
            euler_angles = np.degrees(mat2euler(R_rel, 'sxyz'))  # XYZ convention
            
            # Display relative position
            cv2.putText(frame, f"Relative Position (X,Y,Z):", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"X: {t_rel[0]:.3f}m", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Y: {t_rel[1]:.3f}m", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Z: {t_rel[2]:.3f}m", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display relative orientation
            cv2.putText(frame, f"Relative Orientation (X,Y,Z):", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {euler_angles[0]:.1f}deg", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {euler_angles[1]:.1f}deg", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Yaw: {euler_angles[2]:.1f}deg", (10, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Print to console (for logging/control)
            print(f"\nRelative Position (X,Y,Z): {t_rel[0]:.3f}, {t_rel[1]:.3f}, {t_rel[2]:.3f} meters")
            print(f"Relative Orientation (Roll,Pitch,Yaw): {euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f} degrees")
    
    # Show detection status
    if ids is None or BASE_MARKER_ID not in ids or EE_MARKER_ID not in ids:
        status = "Searching for markers..."
        if ids is not None:
            detected_ids = [id[0] for id in ids]
            status = f"Detected markers: {detected_ids} (need {BASE_MARKER_ID} and {EE_MARKER_ID})"
        cv2.putText(frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Robot End Effector Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()