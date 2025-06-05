import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pickle
from PIL import Image
import numpy as np
import time

# ==== LOAD MODEL ====
print("Loading model....")
model_path = r"C:\Users\ADMIN\Desktop\C files\Dog_breeder\stanford model graphs\mobilenetv2_stanford_raw.pth"
label_encoder_path = r"C:\Users\ADMIN\Desktop\C files\Dog_breeder\stanford model graphs\label_encoder.pkl"

# Load label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

# Load MobileNetV2 (assumes same architecture as training)
from torchvision.models import mobilenet_v2
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))  # adjust for your num classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ==== IMAGE TRANSFORM ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== OPEN WEBCAM ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

starttime = 0
while True:
    currenttime = time.time()
    fps = 1/(currenttime-starttime)
    starttime = currenttime 

    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = transform(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        label = f"{class_names[pred]} {conf.item():.2f}"

    # Draw label (no bbox, just display)
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    cv2.putText(frame, "FPS:" + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("MobileNetV2 Real-time Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
