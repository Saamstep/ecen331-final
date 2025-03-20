import cv2
from ultralytics import YOLO
import numpy as np

# Flag to determine if the frame should be rotated - this was necessary for some of the test videos
rotate_video = False
# Flag to enable disable our lane/edge detecting system
show_lanes = True

# Load the YOLO model
model = YOLO('best.pt') # Custom trained model

# Open the video file
videos = ['pedestrian', 'lights', 'cones']
video_source = cv2.VideoCapture('./archive/train/cones.mov') # Path to input video file

# Function to generate unique colors for each class
def generateColor(class_id):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors
    index = class_id % len(base_colors)
    adjustments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[index][i] + adjustments[index][i] * 
             (class_id // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Helper function to rotate and resize the frame if input is upside down
def adjustFrame(frame):
    # Rotate the frame 90 degrees clockwise
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Adjust the frame to a 16:9 aspect ratio
    height, width = rotated_frame.shape[:2]
    target_width = int(height * 16 / 9)
    resized_frame = cv2.resize(rotated_frame, (target_width, height))
    return resized_frame

def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Hough transform for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Draw lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

while True:
    # Read a frame from the video
    success, frame = video_source.read()
    if not success:
        print("End of video stream or unable to read the frame.")

        break

    # Rotate the frame if the flag is set
    if rotate_video:
        frame = adjustFrame(frame)
    if show_lanes:
        frame = detect_lanes(frame)  # Detect and draw lanes on the frame
    


    # Run the YOLO model on the frame
    detections = model.track(frame, stream=True)

    for detection in detections:
        # Retrieve the class names
        class_labels = detection.names

        # Loop through each detected object
        for box in detection.boxes:
            # Only process objects with confidence above 40%
            if box.conf[0] > 0.4:
                # Extract the bounding box coordinates
                [x_min, y_min, x_max, y_max] = box.xyxy[0]
                # Convert coordinates to integers
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Get the class ID and name
                class_id = int(box.cls[0])
                label = class_labels[class_id]

                # Generate a color for the class
                box_color = generateColor(class_id)

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

                # Add the class label and confidence score to the frame
                cv2.putText(frame, f'{label} {box.conf[0]:.2f}', (x_min, y_min), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
                
    # Display the processed frame
    cv2.imshow('Processed Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
video_source.release()
cv2.destroyAllWindows()