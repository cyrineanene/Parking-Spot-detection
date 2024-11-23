#Creating the data with which we will train the classifier model to identify if the spot is empty or not
import os
import cv2

output_dir = 'data/train'
mask = 'mask/mask_1920_1080.png'

mask = cv2.imread(mask, 0)
analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
totalLabels, label_ids, values, centroid = analysis

slots = []
for i in range (1, totalLabels):
    area = values[i, cv2.CC_STAT_AREA]
    #Coordonnées du point
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]

    #Coordonnées of the bounding box
    pt1 = (x1,y1)
    pt2 = (x1+w, y1+h)
    (X,Y)=centroid[i]

    slots.append([x1, y1, w, h])

video_path = 'data\data_v\parking_1920_1080.mp4'
cap = cv2.VideoCapture(video_path)
frame_number = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        for slot_number, slot in enumerate(slots):
            if slot_number in [132, 147, 165, 180, 344, 360, 377, 385, 341, 360, 179, 131, 106, 91, 61, 4, 98, 129, 161, 185, 201, 224, 271, 303, 319, 335, 351, 389, 29, 12, 32, 72, 281, 280, 157, 223, 26]:
                slot = frame[slot[1]:slot[1] + slot[3], slot[0]:slot[0] + slot[2], :]
                cv2.imwrite(os.path.join(output_dir, '{}_{}.jpg'.format(str(frame_number).zfill(8), str(slot_number).zfill(8))), slot)
        frame_number += 10

cap.release()