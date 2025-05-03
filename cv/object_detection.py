import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from enum import Enum

class DetectionModelType(Enum):
    YOLOv8 = "yolov8"
    FASTER_RCNN = "faster_rcnn"
    MEDIAPIPE = "mediapipe"

class PoseModelType(Enum):
    OPENPOSE = "openpose"
    MEDIAPIPE = "mediapipe"

class HumanDetector:
    def __init__(self, 
                 detection_model_type: DetectionModelType = DetectionModelType.YOLOv8,
                 pose_model_type: PoseModelType = PoseModelType.MEDIAPIPE):
        """
        Initialize the human detector with specified models.
        
        Args:
            detection_model_type: Type of detection model to use
            pose_model_type: Type of pose estimation model to use
        """
        self.detection_model_type = detection_model_type
        self.pose_model_type = pose_model_type
        self.detection_model = self._load_detection_model()
        self.pose_model = self._load_pose_model()
        
        # Robot working frame coordinates (x1, y1, x2, y2)
        # These should be calibrated for your specific setup
        self.working_frame = [0.2, 0.2, 0.8, 0.8]  # Normalized coordinates
        
    def _load_detection_model(self):
        """Load the selected detection model"""
        if self.detection_model_type == DetectionModelType.YOLOv8:
            try:
                from ultralytics import YOLO
                return YOLO('yolov8n.pt')  # Nano version for speed
            except ImportError:
                raise ImportError("Please install ultralytics: pip install ultralytics")
                
        elif self.detection_model_type == DetectionModelType.FASTER_RCNN:
            try:
                import torchvision
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                model.eval()
                return model
            except ImportError:
                raise ImportError("Please install torchvision: pip install torchvision")
                
        elif self.detection_model_type == DetectionModelType.MEDIAPIPE:
            try:
                import mediapipe as mp
                return mp.solutions.object_detection.ObjectDetection()
            except ImportError:
                raise ImportError("Please install mediapipe: pip install mediapipe")
                
        else:
            raise ValueError(f"Unknown detection model type: {self.detection_model_type}")
    
    def _load_pose_model(self):
        """Load the selected pose estimation model"""
        if self.pose_model_type == PoseModelType.OPENPOSE:
            try:
                # This would require OpenPose installation
                raise NotImplementedError("OpenPose implementation requires separate installation")
            except:
                raise ImportError("OpenPose not implemented in this example")
                
        elif self.pose_model_type == PoseModelType.MEDIAPIPE:
            try:
                import mediapipe as mp
                return mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)
            except ImportError:
                raise ImportError("Please install mediapipe: pip install mediapipe")
                
        else:
            raise ValueError(f"Unknown pose model type: {self.pose_model_type}")
    
    def detect_humans(self, image: np.ndarray) -> List[Dict]:
        """Detect humans in the image using the selected model"""
        if self.detection_model_type == DetectionModelType.YOLOv8:
            results = self.detection_model(image)
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # Class 0 is person in YOLO
                        detections.append({
                            'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                            'confidence': box.conf.item()
                        })
            return detections
            
        elif self.detection_model_type == DetectionModelType.FASTER_RCNN:
            import torch
            from torchvision import transforms
            
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(image)
            with torch.no_grad():
                predictions = self.detection_model([img_tensor])
            
            detections = []
            for box, label, score in zip(predictions[0]['boxes'], 
                                       predictions[0]['labels'], 
                                       predictions[0]['scores']):
                if label == 1 and score > 0.5:  # Class 1 is person in COCO
                    detections.append({
                        'bbox': box.cpu().numpy(),
                        'confidence': score.item()
                    })
            return detections
            
        elif self.detection_model_type == DetectionModelType.MEDIAPIPE:
            import mediapipe as mp
            results = self.detection_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detections = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    detections.append({
                        'bbox': np.array([
                            bbox.xmin * image.shape[1],
                            bbox.ymin * image.shape[0],
                            (bbox.xmin + bbox.width) * image.shape[1],
                            (bbox.ymin + bbox.height) * image.shape[0]
                        ]),
                        'confidence': detection.score[0]
                    })
            return detections
            
    def detect_body_parts(self, image: np.ndarray) -> List[Dict]:
        """Detect body parts using pose estimation"""
        if self.pose_model_type == PoseModelType.MEDIAPIPE:
            results = self.pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            body_parts = []
            if results.pose_landmarks:
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    body_parts.append({
                        'id': id,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if hasattr(landmark, 'z') else 0,
                        'visibility': landmark.visibility
                    })
            return body_parts
        else:
            raise NotImplementedError("Only MediaPipe pose estimation is implemented")
    
    def is_in_working_frame(self, x: float, y: float) -> bool:
        """Check if a point is within the robot's working frame"""
        x_norm = x
        y_norm = y
        return (self.working_frame[0] <= x_norm <= self.working_frame[2] and
                self.working_frame[1] <= y_norm <= self.working_frame[3])
    
    def visualize_detections(self, image: np.ndarray, 
                            human_detections: List[Dict], 
                            body_parts: List[Dict]) -> np.ndarray:
        """Visualize detections on the image"""
        vis_image = image.copy()
        
        # Draw working frame
        h, w = image.shape[:2]
        frame_coords = [
            int(self.working_frame[0] * w),
            int(self.working_frame[1] * h),
            int(self.working_frame[2] * w),
            int(self.working_frame[3] * h)
        ]
        cv2.rectangle(vis_image, 
                      (frame_coords[0], frame_coords[1]), 
                      (frame_coords[2], frame_coords[3]), 
                      (0, 255, 0), 2)
        
        # Draw human detections
        for detection in human_detections:
            bbox = detection['bbox'].astype(int)
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(vis_image, f"Person: {detection['confidence']:.2f}", 
                       (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Check if center is in working frame
            center_x = (bbox[0] + bbox[2]) / 2 / w
            center_y = (bbox[1] + bbox[3]) / 2 / h
            if self.is_in_working_frame(center_x, center_y):
                cv2.putText(vis_image, "IN WORKSPACE", 
                           (bbox[0], bbox[1]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw body parts
        for part in body_parts:
            if part['visibility'] > 0.5:  # Only draw visible parts
                x = int(part['x'] * w)
                y = int(part['y'] * h)
                cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
                
                # Check if body part is in working frame
                if self.is_in_working_frame(part['x'], part['y']):
                    cv2.putText(vis_image, f"{part['id']}", 
                               (x+10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_image

    def process_frame(self, image: np.ndarray) -> Tuple[List[Dict], List[Dict], np.ndarray]:
        """Process a single frame and return detections"""
        humans = self.detect_humans(image)
        body_parts = self.detect_body_parts(image)
        visualization = self.visualize_detections(image, humans, body_parts)
        
        # Additional analysis: which detections are in working frame
        for human in humans:
            bbox = human['bbox']
            w, h = image.shape[1], image.shape[0]
            center_x = (bbox[0] + bbox[2]) / 2 / w
            center_y = (bbox[1] + bbox[3]) / 2 / h
            human['in_workspace'] = self.is_in_working_frame(center_x, center_y)
        
        for part in body_parts:
            part['in_workspace'] = self.is_in_working_frame(part['x'], part['y'])
        
        return humans, body_parts, visualization

import cv2
import numpy as np

def main():
    # Initialize detector
    detector = HumanDetector(
        detection_model_type=DetectionModelType.YOLOv8,
        pose_model_type=PoseModelType.MEDIAPIPE
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or video file path
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Set desired resolution (adjust based on your camera capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # For saving output video if needed
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            humans, body_parts, visualization = detector.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('CDPR Safety Monitoring', visualization)
            
            # Save to output video
            # out.write(visualization)
            
            # Print detection info to console
            print(f"\rDetected {len(humans)} humans, {len([p for p in body_parts if p['in_workspace']])} body parts in workspace", end='')
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\nVideo capture released")

if __name__ == "__main__":
    main()