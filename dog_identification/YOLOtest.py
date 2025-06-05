import cv2
import yaml
from ultralytics import YOLO
import time

# Paths (update as needed)
model_path = r"C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\best.pt"
data_yaml_path = r"C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\data.yaml"

# Load YOLO model
model = YOLO(model_path)

# Load class names from YAML
with open(data_yaml_path, 'r') as f:
    data_dict = yaml.safe_load(f)
class_names = data_dict['names']

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

starttime = 0
while True:
    currenttime = time.time()
    fps = 1/(currenttime-starttime)
    starttime = currenttime  

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on frame
    results = model(frame, verbose=False)

    # results is a list (one for each image/frame)
    # Access first result since only one frame at a time
    result = results[0]

    # Draw boxes and labels on frame
    for box in result.boxes:
        # box.xyxy is tensor with coords: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{class_names[cls]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        # Put label text
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Show the frame with detections
    cv2.putText(frame, "FPS:" + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("YOLO Real-time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
