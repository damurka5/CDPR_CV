import cv2
import cv2.aruco as aruco
import numpy as np

# Marker parameters
MARKER_ID = 1  # The specific marker we want to detect
MARKER_LENGTH = 0.05  # 5cm in meters

# Load camera calibration data
calibration_data = np.load('cv/camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Initialize ArUco dictionary and detector
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print(f"Looking for marker ID {MARKER_ID} ({MARKER_LENGTH*100}cm)")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None and MARKER_ID in ids:
        # Find the index of our specific marker
        idx = np.where(ids == MARKER_ID)[0][0]
        
        # Estimate pose for our marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            [corners[idx]], MARKER_LENGTH, camera_matrix, dist_coeffs
        )
        
        # Draw the marker and axis
        frame = aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([MARKER_ID]))
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH/2)
        
        # Extract position and rotation
        tvec = tvec[0][0]  # Translation vector (X, Y, Z)
        rvec = rvec[0][0]  # Rotation vector
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Display information
        cv2.putText(frame, f"Marker {MARKER_ID} Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"X: {tvec[0]:.3f}m", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Y: {tvec[1]:.3f}m", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Z: {tvec[2]:.3f}m", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Print to console (for logging or other processing)
        print(f"\nPosition (X,Y,Z): {tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f} meters")
        print(f"Rotation matrix:\n{rotation_matrix}")
    
    else:
        cv2.putText(frame, f"Searching for marker ID {MARKER_ID}...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("ArUco Marker Localization", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()