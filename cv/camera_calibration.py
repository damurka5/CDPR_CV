import cv2
import numpy as np
import os
from datetime import datetime

# Configuration
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (width, height)
SQUARE_SIZE = 0.025  # Size of one square in meters (25mm)
CALIB_IMAGES_DIR = "calibration_images"
MIN_IMAGES = 15  # Minimum number of calibration images needed

def create_calibration_directory():
    """Create directory for calibration images if it doesn't exist"""
    if not os.path.exists(CALIB_IMAGES_DIR):
        os.makedirs(CALIB_IMAGES_DIR)
        print(f"Created directory: {CALIB_IMAGES_DIR}")
    else:
        print(f"Using existing directory: {CALIB_IMAGES_DIR}")

def capture_calibration_images():
    """Capture calibration images through the camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\n=== Calibration Image Capture ===")
    print("Press:")
    print("  [SPACE] - Capture image when chessboard is detected")
    print("  [q] - Quit when you have enough images (at least 15 recommended)")
    
    image_count = len(os.listdir(CALIB_IMAGES_DIR)) if os.path.exists(CALIB_IMAGES_DIR) else 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # Draw chessboard corners if found
        if ret_chess:
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_chess)
            cv2.putText(frame, "Chessboard detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(frame, f"Captured: {image_count}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE: Capture | q: Quit", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Camera Calibration", frame)
        
        key = cv2.waitKey(1)
        if key == ord(' '):  # SPACE to capture
            if ret_chess:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                img_path = os.path.join(CALIB_IMAGES_DIR, f"calib_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)
                image_count += 1
                print(f"Saved: {img_path}")
                # Flash the screen to indicate capture
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                cv2.addWeighted(flash, 0.3, frame, 0.7, 0, frame)
                cv2.imshow("Camera Calibration", frame)
                cv2.waitKey(300)
        elif key == ord('q'):  # q to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return image_count

def calibrate_camera():
    """Perform camera calibration using captured images"""
    print("\n=== Starting Camera Calibration ===")
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = [os.path.join(CALIB_IMAGES_DIR, f) for f in os.listdir(CALIB_IMAGES_DIR) 
              if f.endswith('.jpg') or f.endswith('.png')]
    
    if len(images) < 5:
        print(f"Error: Need at least 5 images for calibration (found {len(images)})")
        return None, None
    
    print(f"Processing {len(images)} calibration images...")
    
    for i, fname in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}: {fname}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Visualize the corners (optional)
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
        else:
            print(f"Warning: Chessboard not found in {fname}")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < 5:
        print(f"Error: Only found chessboard in {len(objpoints)} images (need at least 5)")
        return None, None
    
    print("\nCalculating camera parameters...")
    
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"\n=== Calibration Results ===")
    print(f"Reprojection error: {mean_error/len(objpoints):.5f} pixels")
    print("(Lower is better, < 0.5 is excellent)")
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    return camera_matrix, dist_coeffs

def save_calibration(filename, camera_matrix, dist_coeffs):
    """Save calibration data to a file"""
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nCalibration data saved to {filename}")

def main():
    create_calibration_directory()
    
    # Step 1: Capture calibration images
    print("\n=== STEP 1: Capture Calibration Images ===")
    print(f"Please prepare a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard pattern")
    print(f"Each square should be {SQUARE_SIZE*1000:.0f}mm in size")
    input("Press Enter to start capturing images...")
    
    image_count = capture_calibration_images()
    print(f"\nCaptured {image_count} calibration images")
    
    if image_count < MIN_IMAGES:
        print(f"Warning: Recommended minimum is {MIN_IMAGES} images")
    
    # Step 2: Perform calibration
    print("\n=== STEP 2: Calibrate Camera ===")
    input("Press Enter to start calibration...")
    
    camera_matrix, dist_coeffs = calibrate_camera()
    
    if camera_matrix is not None:
        # Step 3: Save calibration data
        save_calibration("cv/camera_calibration.npz", camera_matrix, dist_coeffs)
        
        # Step 4: Test calibration
        print("\n=== STEP 3: Test Calibration ===")
        print("Opening camera with undistortion...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Undistort the image
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop the image
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            
            cv2.imshow("Original", frame)
            cv2.imshow("Undistorted", undistorted)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()