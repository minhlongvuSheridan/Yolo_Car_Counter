
# Detection Lines
These are the lines where the counter will count if vehicle pass through it. To create a line, we only two points (x1,y1) and (x2,y2). Finding those points are quite tricky because we need to manually find the best place to put the line. Then estimate it by using the nearest object position. We will create two array of two points with the format [x1, y1, x2, y2]
```python
leftLimits = [180, 230, 310, 230]
rightLimits = [380, 230, 500, 230]
```
we also want to create two counters. It might be weird at first because it is an array. This counter basically store all the id that pass the lines. This is to avoid duplicated Id when it is still within the line in two frames. Then whenver we need to count the total, we simply take the length of the array. This might inefficient since we could explode the memory if there are million of vehicles. But for now, it works
```python
leftCount = []
rightCount = []
```
# Left and right Masks
For whole a image, some objects could be falsely detected from a far (being too small) or from close (being too big). Thus, we would like to have something in the middle. Since we have two roads with two opposite directions I create two masks for right and left. The mask can be create by using the Canva. We basically just cover all the uncessary details with black rectangle.

<img width="640" height="360" alt="right_mask" src="https://github.com/user-attachments/assets/e34f5277-e013-4cc6-bf77-b4d571755ea1" />

*Figure 1. Right mask*

<img width="640" height="360" alt="left_mask" src="https://github.com/user-attachments/assets/59c11ed8-d151-426b-944d-ecf946a226ba" />

*Figure 2. Left mask* <br/>
For the right mask, it is extended until the bottom. It is little tricky because vehicles exist from the very close distance and detection is so sentitve that it could classify the object even with just a small part of vehicle. However, it falsely determine the bounding box of the object, which could mess up our tracking later. Thus by extending it to the bottom, we could minimize the time that wrong bounding box exists. <br/>
Create two mask 
```python
left_mask = cv2.imread("./mask/left_mask.png")
right_mask = cv2.imread("./mask/right_mask.png")
```
Then we create two masked regions *leftRegion* and *rightRegion* by applying the masks to the image. We then use the model on each respective Region and store result into *leftResults* and *rightResults*
```python
leftRegion = cv2.bitwise_and(img, left_mask)
leftResults = model(leftRegion, stream = True)
```
```python
rightRegion = cv2.bitwise_and(img, right_mask)
rightResults = model(rightRegion, stream = True)
```

# Tracking
The problem with normal detection is that it can only detect an object in an image. However, is it still the same object in the immediate next image? To answer this question, 
we will use tracking method which basically assign each detected object an unique identifier.  
Go to https://github.com/abewley/sort, download the sort.py and then import in the CarCounter.py
```python
from sort import *
```
In the main file CarCounter.py, create two tracker objects
```python
leftTracker = Sort(max_age = 20, min_hits = 2,iou_threshold=0.2)
rightTracker = Sort(max_age = 20, min_hits = 2,iou_threshold=0.2)
```
# 
