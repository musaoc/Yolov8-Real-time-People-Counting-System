from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import read_image
import cv2
import imutils
from sort import *
import time

# Load the YOLOv8 model
yolov8_model_path = "yolov8n.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.35,
    device= 'cpu' #"cuda:0" 
)

# Load your video file
video_path = 'Computer Vision task Culture Hint.mp4'
video = cv2.VideoCapture(video_path)

# Initialize the SORT tracker for object tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Define the upper and lower limits for counting objects
limitsEntry = [10, 410, 790, 410]
limitsExit = [10, 370, 790, 370]

# Initialize lists to keep track of counted objects
total_countEntry = []
total_countExit = []

# Iterate over video frames
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Resize the frame
    frame = imutils.resize(frame, width=800, height=480)

    # Increase Contrast and Brightness
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0.5)

    # Perform prediction on the frame
    result = get_prediction(frame, detection_model)

    # Initialize detections array
    detections = np.empty((0, 5))

    for obj in result.object_prediction_list:
        if obj.category.name != "person":
            continue
        bbox = obj.bbox.to_voc_bbox()
        detections = np.vstack((detections, [bbox[0], bbox[1], bbox[2], bbox[3], obj.score.value]))

    # Update the tracker with the new detections
    resultsTracker = tracker.update(detections)

    # Draw bounding boxes and labels on the frame
    for obj in resultsTracker:
        x1, y1, x2, y2, id = obj # Get coordinates and ID
        w, h = x2 - x1, y2 - y1  # Calculate width and height


        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # Draw bounding boxes
        label = f"Person #{id}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Calculate the center of the bounding box
        cx, cy =int(x1 + w // 2), int(y1 + h // 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a circle at the center
        
        # Check if the center of the object crosses the upper limit
        if limitsEntry[0] < cx < limitsEntry[2] and limitsEntry[1] - 15 < cy < limitsEntry[1] + 15:
            if total_countEntry.count(id) == 0:  # Only count if not counted before
                total_countEntry.append(id)  # Add ID to the count list
        
        # Check if the center of the object crosses the lower limit
        if limitsExit[0] < cx < limitsExit[2] and limitsExit[1] - 15 < cy < limitsExit[1] + 15:
            if total_countExit.count(id) == 0:  # Only count if not counted before
                total_countExit.append(id)  # Add ID to the count list
        

        # Draw lines for counting limits
        cv2.line(frame, (limitsEntry[0], limitsEntry[1]), (limitsEntry[2], limitsEntry[3]), (139, 195, 75), thickness=2)
        cv2.line(frame, (limitsExit[0], limitsExit[1]), (limitsExit[2], limitsExit[3]), (50, 50, 230), thickness=2)

        # Draw Heading for Entry and Exit
        cv2.putText(frame, "Entry", (10, 410), cv2.FONT_HERSHEY_PLAIN, 0.7, (139, 195, 75), 2)
        cv2.putText(frame, "Exit", (10, 370), cv2.FONT_HERSHEY_PLAIN, 0.7, (50, 50, 230), 2)


    # Display the total count of objects that crossed the upper and lower limits
    cv2.putText(frame, "ENTRY : " + str(len(total_countEntry)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (139, 195, 75), 2)
    cv2.putText(frame, "EXIT : " +str(len(total_countExit)), (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 230), 2)


    # Display the frame
    cv2.imshow("People Entry/Exit Counting System", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()