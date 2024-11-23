import pickle
from skimage.transform import resize
import numpy as np
import cv2

empty = True
not_empty= False

def empty_or_not(spot_bgr):
    model = pickle.load(open("model/model.p", "rb"))
    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    
    y_output = model.predict(flat_data)

    if y_output == 0:
        return empty
    return not_empty


def get_parking_spots_boxes(connected_components):
    
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1

    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots