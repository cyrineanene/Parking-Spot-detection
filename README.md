# Parking Spot Detection and Availability Counter

This repository contains a solution for detecting parking spots and counting available spots in a parking lot. The system utilizes a Support Vector Classifier to determine whether a parking spot is empty or occupied, based on video input.

## Features

- **Parking Spot Detection:** Detects and classifies parking spots as empty or occupied.
- **Available Spot Counting:** Counts the number of free parking spots in real time.

## Repository Structure

```plaintext
mask/
  ├── mask_1920_1080.png  # Full-sized parking mask for video
  └── mask_crop.png       # Cropped version of the parking 

model/
  ├── classifier.p        # Trained SVC classifier
  ├── crop_cars.py        # Script to prepare training data by cropping car spots from video.
  └── spot_classifier.py  # Main classifier logic

main.py                   # Main script to run parking spot detection and counting
README.md                 
requirements.txt          
utils.py                
```
## Workflow
1. **Data Preparation**:
The dataset is prepared using the `crop_cars.py` script located in the model directory. 

2. **Model Training**:
The Support Vector Classifier (in `spot_classifier.py`) is trained on the processed data to distinguish between empty and occupied spots.

3. **Detection and Counting**:
The script `main.py` performs real-time detection and counts available spots.