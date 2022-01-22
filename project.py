from tkinter import Scale
import cv2

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
    

#initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def click_button(event, x,y ,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    #get frame
    ret, frame =cap.read()

    #object dectection
    (class_ids, score, bboxes)=model.detect(frame)
    for class_id, scores, bbox in zip(class_ids, score, bboxes):
        (x,y,w,h)= bbox
        class_name = classes[class_id]
       

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)

            
        print(x,y,w,h)
        cv2.rectangle(frame,(x,y), (x+w,y+h), (200, 0, 50), 3)
    print("class_ids",class_ids)
    print("score", score)
    print("bboxes", bboxes)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)