import cv2
from utils import get_parking_spots_boxes, empty_or_not
import numpy as np
#import matplotlib.pyplot as plt

def calc_diff(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2)) 

#working on a crop sequence 
# video_path = 'data\data_v\parking_crop_loop.mp4'
# mask = 'mask\mask_crop.png'

#working on the full video
video_path = 'data\data_v\parking_1920_1080_loop.mp4'
mask = 'mask\mask_1920_1080.png'

mask=cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

connected_componets = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) #create a graph relationship between the spots from the mask
parking_spots = get_parking_spots_boxes(connected_componets)

spots_status = [None for j in parking_spots]
diffs = [None for j in parking_spots]
previous_frame = None

ret=True
step = 30
frame_number = 0

while ret:
    ret, frame= cap.read()

    if frame_number % step == 0 and previous_frame is not None: 
        for spot_idx, spot in enumerate(parking_spots):
            x1,y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w]
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w])
    
    #plotting the histogram of the diff between frames
    # plt.hist([diffs[j]/np.amax(diffs) for j in np.argsort(diffs)][::-1] )
    # plt.show()

    if frame_number % step == 0: 
        if previous_frame is None:
            arr = range(len(parking_spots))
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in arr:
            spot = parking_spots[spot_idx]
            x1,y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w]
            #frame = cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (255,0,0))

            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status
    
    if frame_number % step == 0:
        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(parking_spots):
        spot_status = spots_status[spot_idx]
        x1,y1, w, h = parking_spots[spot_idx]
        
        if spot_status:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (0,255,0),2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (0,0,255),2)

    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()