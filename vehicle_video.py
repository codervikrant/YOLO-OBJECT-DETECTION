import cv2
import numpy as np


yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]


#colors = np.random.randit(0, 255, size=(len(classes),3), dtype='uint8' )

colorRed = (0,0,255)
colorGreen = (0,255,0)

# #Loading Images

video = cv2.VideoCapture(0)

if (video.isOpened() == False):  
    print("Error reading video file") 
  
# We need to set resolutions. 
# so, convert them from float to integer. 
width = int(video.get(3)) 
height = int(video.get(4)) 
   
size = (width, height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
result = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
    

while True:
    _, img = video.read()
    #height, width, _ = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)


                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 2)
            cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 2.5, colorRed, 2)
            result.write(img) 

    cv2.imshow("Image", img)
    #cv2.imwrite("output.mp4",img)
    key = cv2.waitKey(500)
    if key==27:
        break

video.release()    
cv2.destroyAllWindows()