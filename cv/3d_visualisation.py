import cv2
import cv2.aruco as aruco
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transforms3d.euler import mat2euler
from time import time
from collections import deque

class CDPRTrackingSystem:
    def __init__(self):
        # Initialize device
        self.device = 'cpu'  # or 'cuda' if available
        
        # ArUco marker configuration
        self.BASE_MARKER_ID = 2
        self.EE_MARKER_ID = 3
        self.MARKER_LENGTH = 0.05  # 5cm
        
        # Load camera calibration
        calibration_data = np.load('cv/camera_calibration.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        
        # Initialize ArUco detector
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, parameters)
        
        # Initialize YOLOv8
        print("Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt").to(self.device)
        
        # Initialize MiDaS for depth estimation
        print("Loading MiDaS model...")
        self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid').to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.dpt_transform
        
        # Base marker pose
        self.base_pose = np.zeros(3)
        
        # Tracking history
        self.ee_positions = deque(maxlen=10)  # Stores (x,y,z) positions
        self.object_positions = {}  # Stores detected objects
        
        # Initialize 3D visualization
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        self.fig.canvas.manager.set_window_title('CDPR 3D Tracking System')
        
        # Performance tracking
        self.last_depth_time = time()
    
    def predict_depth(self, image):
        """Predict depth from RGB image using MiDaS"""
        img_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(img_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 10.0
        return depth_map
    
    def detect_markers(self, frame):
        """Detect ArUco markers and return their poses"""
        corners, ids, _ = self.detector.detectMarkers(frame)
        poses = {}
        
        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], self.MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
                )
                poses[marker_id] = {
                    'rvec': rvec[0][0],
                    'tvec': tvec[0][0],
                    'corners': corners[i]
                }
        
        return poses
    
    def get_relative_pose(self, base_pose, ee_pose):
        """Calculate end effector pose relative to base marker"""
        R_base, _ = cv2.Rodrigues(base_pose['rvec'])
        R_ee, _ = cv2.Rodrigues(ee_pose['rvec'])
        
        self.base_pose = base_pose['tvec']
        
        # Relative transformation
        R_rel = np.dot(R_base.T, R_ee)
        t_rel = np.dot(R_base.T, (ee_pose['tvec'] - base_pose['tvec']))
        
        # Convert to Euler angles
        euler_angles = np.degrees(mat2euler(R_rel, 'sxyz'))
        
        return t_rel, euler_angles
    
    
    def detect_objects(self, frame, depth_map, base_pose=None):
        """Detect objects and estimate their 3D positions relative to base marker"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo(rgb_frame, conf=0.5, device=self.device)
        
        objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = self.yolo.names[int(box.cls)]
                conf = float(box.conf)
                
                # Get 3D position (center of bounding box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                try:
                    z = depth_map[cy, cx]
                    
                    # Convert to 3D camera coordinates
                    fx = self.camera_matrix[0,0]
                    fy = self.camera_matrix[1,1]
                    cx_cam = self.camera_matrix[0,2]
                    cy_cam = self.camera_matrix[1,2]
                    
                    # Camera coordinates (Z forward, Y down, X right)
                    Xc = (cx - cx_cam) * z / fx
                    Yc = (cy - cy_cam) * z / fy
                    Zc = z
                    
                    # Transform to base marker coordinates if available
                    if base_pose is not None:
                        # Get base marker rotation and translation
                        R_base, _ = cv2.Rodrigues(base_pose['rvec'])
                        t_base = base_pose['tvec']
                        
                        # Create homogeneous transformation matrix
                        T_marker_to_cam = np.eye(4)
                        T_marker_to_cam[:3,:3] = R_base
                        T_marker_to_cam[:3,3] = t_base
                        
                        # Transform point from camera to marker coordinates
                        point_cam = np.array([Xc, Yc, Zc, 1])
                        point_marker = np.linalg.inv(T_marker_to_cam) @ point_cam
                        
                        # Swap Y and Z axes to make Z upward
                        Xm = point_marker[0]
                        Ym = -point_marker[2]  # Convert to standard Z-up coordinate system
                        Zm = point_marker[1]
                    else:
                        Xm, Ym, Zm = Xc, -Zc, Yc  # Still swap Y/Z if no base marker
                    
                    objects.append({
                        'label': label,
                        'confidence': conf,
                        'position': np.array([Xm, Ym, Zm]),  # Now in correct coordinate system
                        'bbox': (x1, y1, x2, y2)
                    })
                except:
                    continue
        
        return objects

    def update_3d_visualization(self, ee_position, objects):
        """Update the 3D visualization with correct axes and zoom"""
        self.ax_3d.clear()
        
        # Set the viewing perspective
        self.ax_3d.view_init(elev=20, azim=45)  # Nice viewing angle
        
        # Plot end effector trajectory
        if len(self.ee_positions) > 1:
            positions = np.array(self.ee_positions)
            # Swap Y and Z coordinates for plotting
            self.ax_3d.plot(positions[:,0], positions[:,2], positions[:,1], 
                        'b-', linewidth=2, label='End Effector Path')
        
        # Plot current end effector position
        if ee_position is not None:
            # Swap Y and Z coordinates for plotting
            self.ax_3d.scatter([ee_position[0]], [ee_position[2]], [ee_position[1]], 
                            c='r', s=100, label='End Effector')
        
        # Plot detected objects
        for obj in objects:
            absolute_pos = obj['position']
            relative_pos = absolute_pos - self.base_pose
            pos = [relative_pos[0], relative_pos[2], relative_pos[1]]
            print(f'OBJ POSITION: {[absolute_pos[0]], [absolute_pos[2]], [absolute_pos[1]]}')
            print(f'BASE POSITION: {self.base_pose}')
            
            # print(f'OBJ RELATIVE: {[pos[0]], [pos[2]], [pos[1]]}')
            # Already in correct coordinate system (Z-up)
            self.ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], 
                            c='g', s=50, label=obj['label'])
            self.ax_3d.text(pos[0], pos[1], pos[2], 
                        f"{obj['label']}\n{pos[2]:.2f}m", color='black')
        
        # Set plot limits and labels (zoomed in by 2x)
        self.ax_3d.set_xlim(-0.5, 0.5)  # Reduced from -1,1
        self.ax_3d.set_ylim(-0.5, 0.5)  # Reduced from -1,1
        self.ax_3d.set_zlim(0, 1)       # Reduced from 0,2
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')  # Note we're labeling Y axis as Z now
        self.ax_3d.set_zlabel('Z (m)')  # And Z axis as Y
        self.ax_3d.set_title('3D Tracking View (Z-up)')
        
        # Create custom legend without duplicates
        handles, labels = self.ax_3d.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        self.ax_3d.legend(*zip(*unique))
        
        plt.draw()
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect markers
        marker_poses = self.detect_markers(frame)
        
        # Initialize variables
        ee_position = None
        relative_pose = None
        objects = []
        
        # Process end effector tracking if both markers are detected
        if self.BASE_MARKER_ID in marker_poses and self.EE_MARKER_ID in marker_poses:
            base_pose = marker_poses[self.BASE_MARKER_ID]
            ee_pose = marker_poses[self.EE_MARKER_ID]
            
            # Get relative pose
            t_rel, euler_angles = self.get_relative_pose(base_pose, ee_pose)
            relative_pose = {'position': t_rel, 'orientation': euler_angles}
            ee_position = t_rel
            
            # Store position for trajectory
            self.ee_positions.append(t_rel.copy())
            
            # Draw markers and axes
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                             base_pose['rvec'], base_pose['tvec'], self.MARKER_LENGTH/2)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                             ee_pose['rvec'], ee_pose['tvec'], self.MARKER_LENGTH/2)
            
            # Draw info on frame
            cv2.putText(frame, f"EE Position: X:{t_rel[0]:.2f}m Y:{t_rel[1]:.2f}m Z:{t_rel[2]:.2f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Estimate depth periodically
        current_time = time()
        if current_time - self.last_depth_time > 0.5:
            depth_map = self.predict_depth(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.last_depth_time = current_time
            
            # Pass base_pose if available
            base_pose = marker_poses.get(self.BASE_MARKER_ID)
            objects = self.detect_objects(frame, depth_map, base_pose)
            
            # Update object positions
            for obj in objects:
                if obj['label'] not in self.object_positions:
                    self.object_positions[obj['label']] = deque(maxlen=20)
                self.object_positions[obj['label']].append(obj['position'])
        
        # Update visualization
        self.update_3d_visualization(ee_position, objects)
        # self.update_3d_visualization(ee_position, [])
        
        # Show 2D view
        self.ax_2d.clear()
        self.ax_2d.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.ax_2d.set_title("Camera View")
        self.ax_2d.axis('off')
        
        plt.pause(0.01)
        return frame, ee_position, objects
    
    def run(self, video_source=0):
        """Run the tracking system on video source"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error opening video source {video_source}")
            return
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, _, _ = self.process_frame(frame)
                
                # Display
                cv2.imshow("CDPR Tracking", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()

if __name__ == "__main__":
    print("Initializing CDPR Tracking System...")
    system = CDPRTrackingSystem()
    system.run("cv/videos/webcam_20250508_143816.mp4")  # Or use 0 for webcam