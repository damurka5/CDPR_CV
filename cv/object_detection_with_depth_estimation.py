import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from time import time

# Check for MPS (Apple Silicon) availability
device = 'cpu'#torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DepthDetectionSystem:
    def __init__(self):
        self.device = device
        
        # Initialize YOLOv8 (nano version for speed)
        print("Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt").to(self.device)
        
        # Initialize MiDaS with official transforms
        print("Loading MiDaS model...")
        self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid').to(self.device)
        self.midas.eval()
        
        # Load official MiDaS transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.dpt_transform
        
        # For visualization
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.canvas.manager.set_window_title('CDPR Object + Depth Detection')
        
        # Performance tracking
        self.last_depth_time = time()
    
    def predict_depth(self, image):
        """Predict depth from RGB image using official MiDaS transforms"""
        # Apply official MiDaS transform
        img_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            # MiDaS inference
            prediction = self.midas(img_tensor)
            
            # Resize to original image dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize to 0-10 meters
        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 10.0
        return depth_map
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect objects every frame
                results = self.yolo(rgb_frame, conf=0.5, device=self.device)
                
                # Only estimate depth every 0.5 seconds for better performance
                current_time = time()
                if current_time - self.last_depth_time > 0.5:
                    depth_map = self.predict_depth(rgb_frame)
                    self.last_depth_time = current_time
                
                # Create visualization
                vis_frame = frame.copy()
                
                # Draw detections
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        label = self.yolo.names[int(box.cls)]
                        conf = float(box.conf)
                        
                        # Draw bounding box
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Get depth at center if available
                        if 'depth_map' in locals():
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            try:
                                depth = depth_map[cy, cx]
                                cv2.putText(vis_frame, f"{depth:.2f}m",
                                           (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            except:
                                pass
                        
                        cv2.putText(vis_frame, f"{label} {conf:.2f}",
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Update plots
                self.ax1.clear()
                self.ax2.clear()
                
                self.ax1.imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
                self.ax1.set_title("Object Detection")
                self.ax1.axis('off')
                
                if 'depth_map' in locals():
                    depth_display = np.clip(depth_map, 0, 10)
                    im = self.ax2.imshow(depth_display, cmap='jet', vmin=0, vmax=10)
                    if not hasattr(self, 'cbar'):
                        self.cbar = self.fig.colorbar(im, ax=self.ax2, label='Depth (m)')
                    self.ax2.set_title("MiDaS Depth Estimation")
                    self.ax2.axis('off')
                
                plt.draw()
                plt.pause(0.01)
                
                if plt.waitforbuttonpress(0.01):
                    break
        finally:
            cap.release()
            plt.ioff()
            plt.close()

if __name__ == "__main__":
    print("Initializing CDPR Detection System...")
    system = DepthDetectionSystem()
    system.process_video("cv/videos/webcam_20250508_143816.mp4")