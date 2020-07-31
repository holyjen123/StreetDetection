from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import easydict
import time

video = VideoStream(src=0).start()
fps = FPS().start()

while True:
    time.sleep(2)
  
    image = video.read()
    image = imutils.resize(image, width=800)
    
    args = easydict.EasyDict({
        "prototxt": 'MobileNetSSD_deploy.prototxt',
        "model": 'MobileNetSSD_deploy.caffemodel',
        "confidence": 0.2
    })

    # labels of network
    CLASSES = { 0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

    # loading Caffe model and image
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    height = image.shape[0]/300.0
    width = image.shape[1]/300.0
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    # blob (binary large object)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    frame = image.copy()
    frame1 = image.copy()
    col = cv2.resize(image, (300, 300)).shape[1]
    row = cv2.resize(image, (300, 300)).shape[0]

    # class and location of detected object
    for i in np.arange(0, detections.shape[2]):
        # confidence level and testing
        confidence = detections[0,0,i,2]
        if confidence > args.confidence:
            idx = int(detections[0,0,i,1])

            # object location
            leftX = int(detections[0,0,i,3] * col)
            leftY = int(detections[0,0,i,4] * row)
            rightX = int(detections[0,0,i,5] * col)
            rightY = int(detections[0,0,i,6] * row)

            xLeft = int(width * leftX)
            yLeft = int(height * leftY)
            xRight = int(width * rightX)
            yRight = int(height * rightY)

            cv2.rectangle(image, (xLeft, yLeft), (xRight, yRight), 
                         (0,0,0), 2)

            # creating the image label based off on CLASSES
            if idx in CLASSES:
                label = CLASSES[idx]+ ": " + str(confidence)
                labelSize, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                yLeft = max(yLeft, labelSize[1])
                cv2.rectangle(image, (xLeft, yLeft - labelSize[1]),
                                     (xLeft + labelSize[0], yLeft + base),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xLeft, yLeft),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                
            # center frame: checking the position of the object in image
            if (xRight / imageWidth) < 0.4:
                print('object detected on left')
            elif (xLeft / imageWidth) > 0.6:
                print('object detected on right')
            elif (xLeft / imageWidth) < 0.4 and (xRight / imageWidth) > 0.6:
                if(yLeft / imageHeight) > 0.3:
                    print('warning: object in close range')
                else:
                    print('warning: object really close')
            elif (xLeft / imageWidth) > 0.4 or (xRight / imageWidth) < 0.6:
                if(yLeft / imageHeight) > 0.3:
                    print('warning: object in close range')
                else:
                    print('warning: object really close')    
                    
    # printing results
    cv2.imshow("output", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
    
fps.stop()
cv2.destroyAllWindows()
video.stop()
