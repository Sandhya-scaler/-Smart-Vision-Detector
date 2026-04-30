 import cv2
import numpy as np

# --- 1) Load Haar Cascades ---
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\haarcascade_frontalface_default (1).xml")
eye_cascade = cv2.CascadeClassifier(
    r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\haarcascade_eye (1).xml")
smile_cascade = cv2.CascadeClassifier(
    r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\haarcascade_smile (1).xml")

# --- 2) Load YOLOv3-tiny (FASTER) ---
net = cv2.dnn.readNet(
    r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\yolov3.weights",
    r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\yolov3 (2).cfg"
)
with open(r"C:\Users\Bulbul\OneDrive\Desktop\object_detcter\models_files\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# --- 3) Open webcam ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Press 'q' to quit")

frame_count = 0
previous_detections = []  # store previous boxes
max_memory = 10  # how long to keep old boxes

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- 4) Face detection ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,"face detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # --- Eye detection ---
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(15, 15))
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x+w, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Smile detection ---
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.putText(frame, "Smiling", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- 5) Object detection (every 3rd frame to speed up) ---q
    if frame_count % 3 == 0:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        H, W = frame.shape[:2]

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                wanted_objects = ["bottle", "teddy bear", "book", "chair", "cell phone", "laptop", "cup","banana","mouse"]

                if confidence > 0.5 and classes[class_id] in wanted_objects:
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        new_detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                conf = confidences[i]
                new_detections.append({
                    "box": (x, y, w, h),
                    "label": label,
                    "conf": conf,
                    "age": max_memory
                })

        # Update stored detections
        previous_detections = new_detections

    # --- 6) Draw previous detections ---
    for det in previous_detections:
        x, y, w, h = det["box"]
        label = det["label"]
        conf = det["conf"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        det["age"] -= 1  # reduce life

    # remove old detections
    previous_detections = [d for d in previous_detections if d["age"] > 0]

    frame_count += 1
    cv2.imshow("Smart Vision Detector (Smooth Detection)", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
