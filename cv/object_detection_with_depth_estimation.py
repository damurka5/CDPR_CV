import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.transforms import Normalize

class SimpleDepthEstimator(nn.Module):
    """A simplified depth estimation model that works with the architecture"""
    def __init__(self):
        super(SimpleDepthEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class DepthDetectionSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLO
        print("Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt")
        
        # Initialize depth estimator
        print("Loading depth estimation model...")
        self.depth_model = SimpleDepthEstimator().to(self.device)
        self.depth_model.eval()
        
        # Normalization
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        
    def predict_depth(self, image):
        """Predict depth from RGB image"""
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).to(self.device)
        img_tensor = self.normalize(img_tensor / 255.0).unsqueeze(0)
        
        with torch.no_grad():
            depth = self.depth_model(img_tensor)
            return depth.squeeze().cpu().numpy()
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            results = self.yolo(rgb_frame, conf=0.5)
            
            # Estimate depth
            depth_map = self.predict_depth(rgb_frame)
            depth_map = (depth_map * 255).astype(np.uint8)  # Scale for visualization
            
            # Draw detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    label = self.yolo.names[int(box.cls)]
                    conf = float(box.conf)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Get depth at center
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = depth_map[cy, cx] / 255.0 * 10.0  # Scale back to meters
                    
                    # Display info
                    cv2.putText(frame, 
                               f"{label} {conf:.2f}",
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame,
                               f"{depth:.2f}m",
                               (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display depth map
            depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            # Combine frames
            combined = np.hstack((frame, depth_colormap))
            
            cv2.imshow("Object Detection + Depth Estimation", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = DepthDetectionSystem()
    system.process_video("cv/videos/webcam_20250508_143816.mp4")