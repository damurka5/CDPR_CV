import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import os
from timm.models.layers import trunc_normal_
from timm.models.efficientnet import tf_efficientnet_b5_ap
from torchvision.transforms import Normalize

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU()
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)
        
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class mViT(nn.Module):
    """Mini Vision Transformer for adaptive bins"""
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256, 
                 embedding_dim=128, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_size = patch_size
        
        self.conv_input = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1, padding=0)
        
        # Transformer layers would go here (simplified for this implementation)
        self.proj_out = nn.Linear(embedding_dim, dim_out)
        
    def forward(self, x):
        # Simplified implementation - replace with actual transformer layers if needed
        x = self.conv_input(x)
        # Global average pooling instead of transformer for this example
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        bin_widths_normed = torch.sigmoid(self.proj_out(x))
        range_attention_maps = torch.ones_like(x[:, :1])  # Dummy attention
        return bin_widths_normed, range_attention_maps

class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=256, min_val=0.1, max_val=10.0, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                      dim_out=n_bins, embedding_dim=128, norm=norm)
        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        unet_out = self.decoder(self.encoder(x))
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)
        
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return pred.squeeze(1)  # Return just the depth map

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel = tf_efficientnet_b5_ap(pretrained=True)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        return cls(basemodel, n_bins=n_bins, **kwargs)

class DepthEstimator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.min_depth = 0.1
        self.max_depth = 10.0
        self.model = UnetAdaptiveBins.build(256, min_val=self.min_depth, max_val=self.max_depth).to(self.device)
        
        # Load state dict with strict=False to handle missing mViT components
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Normalization parameters
        self.normalize = torch.nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def predict(self, image):
        """Predict depth from RGB image"""
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).to(self.device)
        img_tensor = self.normalize(img_tensor / 255.0).unsqueeze(0)
        
        with torch.no_grad():
            depth = self.model(img_tensor)
            return depth.squeeze().cpu().numpy()

class CollisionAvoidanceSystem:
    def __init__(self):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuration
        self.ARUCO_DICT = cv2.aruco.DICT_6X6_250
        self.EE_MARKER_ID = 3
        self.MARKER_LENGTH = 0.05
        self.OBJECT_COLOR = "red"
        self.MIN_CONFIDENCE = 0.7
        
        # Initialize models
        print("Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt")
        
        print("Loading AdaBins model...")
        self.depth_estimator = DepthEstimator("AdaBins/pretrained/AdaBins_nyu.pt", self.device)
        
        # Initialize ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Load camera calibration
        calibration_data = np.load('cv/camera_calibration.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        
        # Visualization
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        plt.ion()
        
        # Tracking
        self.object_positions = deque(maxlen=20)
        self.ee_positions = deque(maxlen=20)
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame
                continue
                
            # Detect end effector
            ee_pos, ee_rot = self.detect_end_effector(frame)
            
            # Detect objects and estimate depth
            objects = self.detect_objects(frame)
            objects_with_depth = self.estimate_object_depth(frame, objects)
            
            # Filter red objects
            # red_objects = self.filter_red_objects(frame, objects_with_depth)
            
            # Update visualization
            # self.update_3d_visualization(ee_pos, red_objects)
            self.update_3d_visualization(ee_pos, objects_with_depth)
            
            # Show processing result
            # vis_frame = self.create_visualization_frame(frame, ee_pos, red_objects)
            vis_frame = self.create_visualization_frame(frame, ee_pos, objects_with_depth)
            cv2.imshow("Processing", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
    
    def detect_end_effector(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None and self.EE_MARKER_ID in ids:
            idx = np.where(ids == self.EE_MARKER_ID)[0][0]
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], self.MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
            )
            return tvec[0][0], rvec[0][0]
        return None, None
    
    def detect_objects(self, frame):
        results = self.yolo(frame, conf=self.MIN_CONFIDENCE)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = self.yolo.names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                
                if label.lower() in ['book', 'cell phone', 'remote']:
                    continue
                    
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox,
                    'class_id': class_id
                })
        return detections
    
    def estimate_object_depth(self, frame, objects):
        if not objects:
            return []
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map = self.depth_estimator.predict(rgb_frame)
        
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            obj_region = depth_map[y1:y2, x1:x2]
            obj['depth'] = np.median(obj_region)
            
        return objects
    
    def filter_red_objects(self, frame, objects):
        red_objects = []
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            obj_region = hsv_frame[y1:y2, x1:x2]
            
            mask1 = cv2.inRange(obj_region, lower_red1, upper_red1)
            mask2 = cv2.inRange(obj_region, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            red_pixels = cv2.countNonZero(red_mask)
            total_pixels = (x2 - x1) * (y2 - y1)
            red_ratio = red_pixels / total_pixels
            
            if red_ratio > 0.3:
                obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                depth = obj['depth']
                
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                
                x = (obj_center[0] - cx) * depth / fx
                y = (obj_center[1] - cy) * depth / fy
                z = depth
                
                obj['position'] = np.array([x, y, z])
                red_objects.append(obj)
                
        return red_objects
    
    def update_3d_visualization(self, ee_pos, objects):
        self.ax.clear()
        
        if ee_pos is not None:
            self.ee_positions.append(ee_pos)
            if len(self.ee_positions) > 1:
                ee_pos_array = np.array(self.ee_positions)
                self.ax.plot(ee_pos_array[:, 0], ee_pos_array[:, 1], ee_pos_array[:, 2], 
                            'b-', label='End Effector Path')
            self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], 
                           c='blue', marker='o', s=100, label='End Effector')
        
        for obj in objects:
            pos = obj['position']
            self.object_positions.append(pos)
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                           c='red', marker='x', s=100, label='Obstacle')
        
        self.ax.legend()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        plt.draw()
        plt.pause(0.001)
    
    def create_visualization_frame(self, frame, ee_pos, objects):
        vis_frame = frame.copy()
        
        if ee_pos is not None:
            cv2.putText(vis_frame, f"EE Position: {ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            pos = obj['position']
            cv2.putText(vis_frame, f"{obj['label']}: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}m", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_frame

if __name__ == "__main__":
    print("Initializing Collision Avoidance System...")
    system = CollisionAvoidanceSystem()
    
    # Download AdaBins weights if not present
    # if not os.path.exists("AdaBins_nyu.pt"):
    #     print("Downloading AdaBins weights...")
    #     import urllib.request
    #     urllib.request.urlretrieve(
    #         "https://github.com/shariqfarooq123/AdaBins/releases/download/v1.0/AdaBins_nyu.pt",
    #         "AdaBins_nyu.pt"
    #     )
    
    system.process_video("cv/videos/webcam_20250508_143816.mp4")  # Change to your video path