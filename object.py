import cv2

img = cv2.imread("car,human.png")

# convert color image to gray scale
imgGray = cv2.imread('car,human.png', 0)

car_classifier = cv2.CascadeClassifier("cars.xml")
pedestrian_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

cars=  car_classifier.detectMultiScale(imgGray, 1.1, 1)
pedestrian = pedestrian_classifier.detectMultiScale(imgGray,1.1,1)

thresh=0.5

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(configPath,weightsPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, conf, bbox =net.detect(img,thresh)

for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img,"Car", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

for (x, y, w, h) in pedestrian:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, "Human", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
for confidence,box in zip(conf.flatten(),bbox):
    cv2.putText(img, str(round(confidence*100,2)),(box[0]+30,box[1]+30) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

# Display frames in a window
cv2.imshow('Detection', img)
cv2.waitKey(0)


cv2.destroyAllWindows()
