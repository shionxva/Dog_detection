import cv2
import torch
import pickle
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import time

# ========== Load COCO Class Names ==========
print("loading 1st model")
detect_labels = r'C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\coco.names'
detect_model = r'C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
with open(detect_labels, "r") as f:
    coco_classes = [line.strip() for line in f.readlines()]
dog_class_id = coco_classes.index("dog")  # usually 17

# ========== Load OpenCV DNN SSD Model ==========
frozen_inference_graph = r'C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(frozen_inference_graph, detect_model)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ========== Load MobileNetV2 Classifier ==========
print('loading 2nd model')
label_encoder_path = r"C:\Users\ADMIN\Desktop\C files\Dog_breeder\stanford model graphs\label_encoder.pkl"
model_path = r"C:\Users\ADMIN\Desktop\C files\Dog_breeder\stanford model graphs\mobilenetv2_stanford_raw.pth"

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ========== Image Transform ==========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Start Webcam ==========
cap = cv2.VideoCapture(0)
starttime = 0

while True:
    currenttime = time.time()
    fps = 1 / (currenttime - starttime)
    starttime = currenttime

    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5)

    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            if class_id != dog_class_id:
                continue  # skip non-dog detections

            x, y, w, h = box
            dog_crop = frame[y:y+h, x:x+w]

            if dog_crop.size == 0:
                continue

            input_tensor = transform(dog_crop).unsqueeze(0)
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                pred = torch.argmax(probs, 1).item()
                breed = label_encoder.inverse_transform([pred])[0]

            # Draw box and breed name
            label = f"{breed}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 30), (x + len(label) * 12, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Dog Detector + Breed Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
