'''
Author: Minh Long Vu
Application: Inspired by the tutorial of https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=7189s. I develope the YOLO model further
by detecting two roads instead of one
'''

from ultralytics import YOLO
import cv2
from sort import *
from helper import *



# yolov11 specify the version and l specify the weight 
# more weight more accurate and detailed but also slower

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 360))

model = YOLO('yolo11l.pt')
model_names = model.names
# store the id that cross the line
leftCount = []
rightCount = []

cap = cv2.VideoCapture("./video/cars1.mp4")
# the mask is created by using canva where you cover all unnessary details using
# a black box
left_mask = cv2.imread("./mask/left_mask.png")
right_mask = cv2.imread("./mask/right_mask.png")
leftLimits = [180, 230, 310, 230]
rightLimits = [380, 230, 500, 230]

# tracking
# max_age: maximum numnber of frames the we still recognize it
leftTracker = Sort(max_age = 20, min_hits = 2,iou_threshold=0.2)
rightTracker = Sort(max_age = 20, min_hits = 2,iou_threshold=0.2)

while True:
    success, img = cap.read()
    
    #### Right Region
    leftRegion = cv2.bitwise_and(img, left_mask)
    leftResults = model(leftRegion, stream = True)
    leftDetections = add_class_to_detections(leftResults, model_names)
    print(f"outisde: {leftDetections}")
    left_tracker_results = leftTracker.update(leftDetections)
    # draw the line
    cv2.line(img, (leftLimits[0],leftLimits[1]),(leftLimits[2],leftLimits[3]),(0,0,255),5)
    leftCount = track_passing_lines(img, leftLimits, left_tracker_results, leftCount)
    # draw count
    cv2.putText(img, f"Left Count: {len(leftCount)}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 2)
    
    
    #### Right Region
    rightRegion = cv2.bitwise_and(img, right_mask)
    rightResults = model(rightRegion, stream = True)
    rightDetections = add_class_to_detections(rightResults, model_names)
    right_tracker_results = rightTracker.update(rightDetections)
    cv2.line(img, (rightLimits[0],rightLimits[1]),(rightLimits[2],rightLimits[3]),(0,0,255),5)
    rightCount = track_passing_lines(img, rightLimits, right_tracker_results, rightCount)
    cv2.putText(img, f"Right Count: {len(rightCount)}",(400,320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 2)
    
    
    cv2.putText(img, "Minh Long Vu",(0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (1, 50, 32), 1)
    # cv2.imshow("Image", img)
    cv2.imwrite("./video/demo.mp4",img)
    out.write(img)
    cv2.waitKey(1)




    
