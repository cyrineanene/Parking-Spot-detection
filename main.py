import cv2
from utils import get_parking_spots_boxes

#working on a crop sequence 
video_path = 'data\data_v\parking_crop_loop.mp4'
mask = 'data\mask_crop.png'

mask=cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_componets = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) #create a graph relationship between the spots from the mask
parking_spots = get_parking_spots_boxes(connected_componets)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

ret=True
while ret:
    ret, frame= cap.read()
    
    for spot in parking_spots:
        x1,y1, w, h = spot
        frame = cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (255,0,0))

    

    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()