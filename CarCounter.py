from ultralytics import YOLO
import cv2
import math

# yolov8 specify the version and l specify the weight 
# more weight more accurate and detailed but also slower
model = YOLO('yolov8l.pt')
model_names = model.names


cap = cv2.VideoCapture("./video/cars1.mp4")
# the mask is created by using canva where you cover all unnessary details using
# a black box
left_mask = cv2.imread("./mask/left_mask.png")


while True:
    success, img = cap.read()
    
    imgRegion = cv2.bitwise_and(img, left_mask)
    
    # stream = True use generator which is more efficient
    results = model(imgRegion, stream = True)
    # the model always return list of Resutls 
    # each Results is basically an image coressponding with input image
    # each Results contains detect objects which are bounding bosex
    for result in results:
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
            
            if currentClass == "car":
                # Draw the box
                cv2.rectangle(img,(x1,y1), (x2,y2),(13, 13, 13), 1)
                # Write the class name and confidence percentage
                cv2.putText(img, f"{currentClass}: {conf}%",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 255, 51), 1)
            
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)


# show will display in seperate window, we are having trouble with cv2.waitKey() since it doesn't work
# so we will use save to save image



