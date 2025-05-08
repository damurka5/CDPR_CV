import cv2
import datetime

# Video capture setup
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get camera frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # You can adjust this or get it from the camera with cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# FourCC is a 4-byte code used to specify the video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for AVI format
output_filename = f"cv/webcam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

# Create VideoWriter object
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

print(f"Recording started. Saving to {output_filename}")
print("Press 'q' to stop recording...")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Write the frame to the output file
    out.write(frame)
    
    # Display the frame (optional)
    cv2.imshow('Recording...', frame)
    
    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Recording saved to {output_filename}")