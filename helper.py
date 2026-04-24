'''
Author: Minh Long Vu
This file store the helper functions for the main program
'''

import cv2
import math
from sort import *


def add_class_to_detections(Results, names):
    
    '''
    Results:
    names: Array names of the classes.
    '''
    
    
    Detections = np.empty((0, 5))
    for result in Results:
        # 2D tensor of detected bouding box
        boxes = result.boxes
        for box in boxes: 
            # method to get points
            # x1,y1 is the top left corner
            # x2, y2 is the boottom right corner
            x1, y1, x2, y2 = box.xyxy[0]
            
            # the result is something like tensor(123) so we need to convert it to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = math.ceil(box.conf[0] * 100) 
            # box.cls[0] only give us the id. We need to map it with actual class
            # each model might have varying class name, thus we will use model.names to get all the name
            currentClass = names[int(box.cls[0])]
            
            if (currentClass == "car" or currentClass == "truck" or currentClass == "bus") and conf > 50:
                currentArray = np.array([x1, y1, x2, y2, conf])
                Detections = np.vstack((Detections, currentArray))
                print(f"inside: {Detections}")
    return Detections

def track_passing_lines(img, Limits, tracker_results, total_count):
    
    '''
    img: the image
    Limits: Specify the limit line .It is array of [x1, y1, x2, y2] where (x1,y1) and
    (x2,y2) are the first point and second point, respectively 
    tracker_results:
    total_count: An array that store all the id. 
    '''

    
    for tracker_result in tracker_results:
        x1, y1, x2, y2, id = tracker_result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        cv2.rectangle(img,(x1,y1), (x2,y2),(13, 13, 13), 1)
        cv2.putText(img, f"id: {id}",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 255, 51), 1)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (center_x, center_y), 3, (255,0,255), 1)
        if Limits[0] - 5 < center_x < Limits[2] + 5 and  Limits[1] - 5 < center_y < Limits[3] + 5:
            if(total_count.count(id) == 0):
                total_count.append(id)    
    return total_count