import cv2 as cv
import os
import numpy as np
import time
#import torch




#toggle the dev to switch the print statments
dev = 1

#Const paths to models, labels and weights trained with yolov4 path
pathToWeights = 'basketball/weights/best.weights'
pathToModel = 'basketball/cfg/basketball.cfg'
pathToLabels = 'basketball/data/basketball.names'
pathToVideo = 'assests/test_bb_26557.mp4'

scale_percent = 40 # percent of original size


confidence_threshold = 0.5
non_maxima_threshold = 0.3

#Load the trained labels
LABELS = open(pathToLabels).read().strip().split('\n')

# initialize a list of colors to represent each possible action label
np.random.seed(69)
COLOR = np.random.randint(0,255,size=(len(LABELS),3),dtype='uint8')

if dev:
    print((LABELS))

#Load the yolo net with the cv2 DNN , get the layer 82, 94 and 106 which have the predictions
net = cv.dnn.readNetFromDarknet(pathToModel,pathToWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i[0] - 1 ] for i in net.getUnconnectedOutLayers()]
if dev:
    print(ln)

#Initialize the video steam for reading the video file
source = cv.VideoCapture(pathToVideo)
writer = None
(W,H) = (None,None)

#Find the total frames and processing time for the vides
try:
    total_Frames = int(source.get(cv.CAP_PROP_FRAME_COUNT))
    if dev:
        print(f'[INFO] {total_Frames} total frame in the video')

except:
    print("[ERROR unable to find the total frame count")
    total_Frames = -1


# Start of the loop grab the frame , pass it thru the net, gather predictions, apply non maximal averaging and translate
# the W and H to the image

while True:

    (didRet, frame) = source.read()

    dim = (512, 512)

    #no return ==> end of frame , break
    if not didRet:
        break
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    if W is None or H is None:
        (H,W) = frame.shape[:2]

    #Preparing the source video to be passed into the net for a forward pass
    blob = cv.dnn.blobFromImage(frame,1/255.0,(416,416), swapRB=True,crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    # initialize  lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    #loop over each of the output layers from the net
    for output in layerOutputs:

        #loop over each detection
        for detection in output:
            #extract the classID and confidence of the current detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            #setting the thresholds for the detections
            if confidence > confidence_threshold:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX,centerY, width,height) = box.astype("int")

                #cordinates of the top left corner
                x = int(centerX -(width/2))
                y = int(centerY - (height/2))

                #collate the detections into lists
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                # Apply the non-maxima supression using the NMSBoxes to reduce the number of boxes predicted to a prominent one
        idxs = cv.dnn.NMSBoxes(boxes,confidences,confidence_threshold,non_maxima_threshold)

        # ensure we have at least one prediction
        if len(idxs) > 0:
            # loop over the indxes
            for i in idxs.flatten():
                if i > len(COLOR):
                    print('[INFO] Done !!!')
                    break
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw the bounding box
                color = [int(c) for c in COLOR[classIDs[i]]]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, f"{LABELS[classIDs[i]]}:{confidences[i]}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           color, 2)
                print(f'Detected :{LABELS[classIDs[i]]} at  {int(source.get(cv.CAP_PROP_POS_MSEC))}, point = ({x,y})')

        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter('output/test.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            if total_Frames > 0:
                elap = (end - start)
                print(f'[INFO] single frame took {elap}')

        writer.write(frame)



print("[INFO] cleaning up")
source.release()
writer.release()
cv.destroyAllWindows()