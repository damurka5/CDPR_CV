import cv2
import time

def open_camera():
    """Try different methods to open the webcam"""
    # Try different backends and camera indices
    backends = [
        cv2.CAP_DSHOW,       # Windows DirectShow
        cv2.CAP_MSMF,        # Windows Media Foundation
        cv2.CAP_V4L2,        # Linux
        cv2.CAP_ANY          # Auto-detect
    ]
    
    for index in [0, 1, 2]:  # Try different camera indices
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            time.sleep(0.5)  # Give camera time to initialize
            if cap.isOpened():
                print(f"Successfully opened camera index {index} with backend {backend}")
                return cap
            cap.release()
    
    print("Error: Could not open any camera")
    return None

def main():
    cap = open_camera()
    if cap is None:
        return
    
    try:
        # Try to disable autofocus
        autofocus_props = [
            cv2.CAP_PROP_AUTOFOCUS,
            cv2.CAP_PROP_FOCUS,
            getattr(cv2, 'CAP_PROP_AUTO_FOCUS', None),  # Some cameras use this
        ]
        
        for prop in autofocus_props:
            if prop is not None:
                try:
                    if cap.set(prop, 0):
                        print(f"Disabled autofocus using property {prop}")
                    else:
                        print(f"Failed to set property {prop} (may be read-only)")
                except:
                    print(f"Error setting property {prop}")
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  'f' - Try to set focus manually")
        
        focus_value = 50
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
                
            cv2.imshow('Webcam Feed', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):
                focus_value = (focus_value + 10) % 100
                if cap.set(cv2.CAP_PROP_FOCUS, focus_value):
                    print(f"Set focus to {focus_value}")
                else:
                    print("Failed to set focus")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    main()