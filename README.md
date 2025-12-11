🎥 Smart Vision Detector

A powerful real-time face, eyes, smile, and object detection system built using:

🧠 OpenCV Haar Cascades (Face, Eyes, Smile)

🚀 YOLOv3-Tiny (fast object detection)

✔️ Smooth detection using tracking and memory

This app gives fast, accurate, and stable detections using a webcam.

✨ Features
👤 Face Detection

Detects frontal face

Draws bounding box

Shows label “Face Detected”

👀 Eyes Detection

Works inside face ROI

Prints “Eyes Detected” when found

😄 Smile Detection

Detects smile inside face ROI

Displays “Smiling”

📦 YOLO Object Detection

The following objects are detected with YOLOv3-tiny:

Object
Bottle
Teddy Bear
Book
Chair
Cell Phone
Laptop
Cup
Banana
Mouse
🔥 YOLO Features

Detects every 3rd frame → 3× faster

Non-Max Suppression (NMS) for clean boxes

Smooth tracking with box memory

Removes jitter with age-based decay

🧰 Technologies Used
Library	Purpose
OpenCV	Video processing, Haar cascades, drawing
NumPy	Array calculations
YOLOv3-tiny (Darknet)	Fast object detection
Cascade Classifiers	Face/Eyes/Smile detection
📂 Project Files Used

Place your model files inside models_files/:

File	Purpose
haarcascade_frontalface_default.xml	Face detection
haarcascade_eye.xml	Eye detection
haarcascade_smile.xml	Smile detection
yolov3.weights	YOLO weights
yolov3.cfg	YOLO model
coco.names	List of object categories
🛠️ Installation

Install required packages:

pip install opencv-python numpy

▶️ How to Run

Run the script:

python smart_vision_detector.py


Press Q anytime to quit.

🧠 How It Works (Logic Breakdown)
1️⃣ Haar Cascade Detection

Convert frame to grayscale

Detect face → draw box

Detect eyes inside face → show label

Detect smile → show "Smiling"

2️⃣ YOLO Object Detection

Every 3rd frame:

Convert frame to blob

Forward pass through YOLO

Filter only required objects

Use NMS to clean results

3️⃣ Smooth Tracking

To avoid flicker:

Detected boxes are stored for 10 frames

Each box has "age"

Slowly disappears → stable experience

New detections refresh the age

4️⃣ Display

Shows all detections beautifully on the webcam window.

🔧 Customize YOLO Objects

You can edit this list:

wanted_objects = [
    "bottle", "teddy bear", "book", "chair", 
    "cell phone", "laptop", "cup", "banana", "mouse"
]


Add any object from coco.names.

📸 Output Example (You Can Add Screenshot)
-----------------------------------------
[ Face Detected ]   [ Eyes Detected ]  
[ Smiling ]  

Bottle 0.87
Laptop 0.92
Cup 0.78
-----------------------------------------

🚀 Future Improvements

Face mask detection

Age & gender prediction

Object tracking with DeepSORT

Emotion detection

Add sound alert for special detections