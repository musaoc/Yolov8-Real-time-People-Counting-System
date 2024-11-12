# People Entry/Exit Counting System Using YOLOv8 and SORT Tracker

## Overview
This repository contains a Python-based program that detects and tracks people in a video, counting the number of individuals entering and exiting a defined area. It uses the YOLOv8 model for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking.

## Prerequisites
Before running the code, make sure the following Python libraries are installed:

- **sahi**: For seamless integration with the YOLO model.
- **OpenCV (cv2)**: For video frame manipulation and display.
- **imutils**: For frame resizing.
- **numpy**: For numerical operations.
- **SORT**: For tracking detected objects across video frames.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/entry-exit-counter.git
   cd entry-exit-counter

2. **Create and activate a virtual environment (optional)**:
```
  python -m venv env
  source env/bin/activate  # On Windows, use "env\Scripts\activate"
```

3. **Install dependencies**:
```
  pip install sahi opencv-python imutils numpy
```
4. **Run Program**:
```
  python entry_exit_counter.py
```

## Features
- Real-time detection and tracking using YOLOv8 and SORT.
- Entry and exit counting with customizable boundaries.
- Graphical display with bounding boxes and unique IDs for detected individuals.
- High contrast video processing for better visibility

## Configuration
- video_path: Set the path to your video file in the script.
- confidence_threshold: Adjust this to change detection confidence.
- iou_threshold: Modify this in the SORT tracker for performance tuning.
- device: Set to 'cuda:0' for GPU processing or 'cpu' for CPU processing.

## System Requirements
- Python 3.6+
- Compatible with both CPU and GPU execution (GPU recommended for better performance).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or additions.

## Acknowledgments
- Ultralytics for the YOLOv8 model.
- SORT algorithm by Alex Bewley et al.
- SAHI
