# OpenCV For Autonomous Vehicle Systems

# Files Explained
`custom.yaml` - Custom configuration for YOLO custom dataset training

`prepare_datasets.py` - Preprocessing work to setup and combine datasets

`yolo.ipybn` - Training of the custom model using gathered datasets

`webcam_test.py` - Exploratory introduction into using OpenCV and manipulating frames by rotating and resizing webcam feed

`yolo_opencv_implementation` - System to run OpenCV with custom YOLO model
* **NOTE:** The trained model `best.pt` must be in the root project directory
* Path to input videos relative to root project directory
* Enable/Disable Settings at the top of the Python file
  * `rotate_video`: Flag to determine if the frame should be rotated - this was necessary for some of the test videos
  * `show_lanes`: Flag to enable disable our lane/edge detecting system

# Setup
```bash
git clone https://github.com/Saamstep/ecen331-final opencv-yolo
cd opencv-yolo
pip install -r requirements.txt
```

# Thank You
* Lab 06
* https://codezup.com/computer-vision-autonomous-vehicles/
* OpenCV Documentation
* Datasets
  * https://public.roboflow.com/object-detection/pothole
  * https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking
  * https://universe.roboflow.com/ethan-carlson/conetest
  * https://public.roboflow.com/object-detection/self-driving-car
