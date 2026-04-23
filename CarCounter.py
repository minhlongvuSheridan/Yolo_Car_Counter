'''
Author: Minh Long Vu
Application: Inspired by the tutorial of https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=7189s. I develope the YOLO model further
by detecting two roads instead of one
'''



from ultralytics import YOLO
import cv2
import math
from sort import *

# yolov8 specify the version and l specify the weight 
# more weight more accurate and detailed but also slower
model = YOLO('yolov8l.pt')
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
    
    leftRegion = cv2.bitwise_and(img, left_mask)
    rightRegion = cv2.bitwise_and(img, right_mask)
    
    # stream = True use generator which is more efficient
    leftResult = model(leftRegion, stream = True)
    rightResult = model(rightRegion, stream = True)
    # the model always return list of Resutls 
    # each leftResult is basically an image coressponding with input image
    # each leftResult contains detect objects which are bounding bosex
    
    #### Left Region
    leftDetections = np.empty((0, 5))
    for result in leftResult:
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
            currentClass = model_names[int(box.cls[0])]
            
   
            
            
            if (currentClass == "car" or currentClass == "truck" or currentClass == "bus") and conf > 50:
                currentArray = np.array([x1, y1, x2, y2, conf])
                leftDetections = np.vstack((leftDetections, currentArray))
    leftResults = leftTracker.update(leftDetections)
    
    # draw the line
    cv2.line(img, (leftLimits[0],leftLimits[1]),(leftLimits[2],leftLimits[3]),(0,0,255),5)
    for leftResult in leftResults:
        x1, y1, x2, y2, id = leftResult
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(f"Left: {leftResult}")
        cv2.rectangle(img,(x1,y1), (x2,y2),(13, 13, 13), 1)
        # Write the class name and confidence percentage
        cv2.putText(img, f"{id}",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 255, 51), 1)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (center_x, center_y), 3, (255,0,255), 1)
        if leftLimits[0] - 5 < center_x < leftLimits[2] + 5 and  leftLimits[1] - 5 < center_y < leftLimits[3] + 5:
            if(leftCount.count(id) == 0):
                leftCount.append(id)    
    cv2.putText(img, f"Left Count: {len(leftCount)}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 2)
    
    
    
    #### Right Region
    rightDetections = np.empty((0, 5))
    for result in rightResult:
        boxes = result.boxes
        for box in boxes: 
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(box.conf[0] * 100) 
            currentClass = model_names[int(box.cls[0])]   
            if (currentClass == "car" or currentClass == "truck" or currentClass == "bus") and conf > 50:
                currentArray = np.array([x1, y1, x2, y2, conf])
                rightDetections = np.vstack((rightDetections, currentArray))
    rightResults = rightTracker.update(rightDetections)
    
    cv2.line(img, (rightLimits[0],rightLimits[1]),(rightLimits[2],rightLimits[3]),(0,0,255),5)
    for rightResult in rightResults:
        x1, y1, x2, y2, id = rightResult
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(f"Right: {rightResult}")
        cv2.rectangle(img,(x1,y1), (x2,y2),(13, 13, 13), 1)
        # Write the class name and confidence percentage
        cv2.putText(img, f"{id}",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 255, 51), 1)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (center_x, center_y), 3, (255,0,255), 1)
        if rightLimits[0] - 5 < center_x < rightLimits[2] + 5 and  rightLimits[1] - 5 < center_y < rightLimits[3] + 5:
            if(rightCount.count(id) == 0):
                rightCount.append(id)    
    cv2.putText(img, f"Right Count: {len(rightCount)}",(400,320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 2)
    cv2.imshow("Image", img)
    cv2.imshow("Image2", rightRegion)
    cv2.waitKey(1)


# show will display in seperate window, we are having trouble with cv2.waitKey() since it doesn't work
# so we will use save to save image



