import cv2 as cv
import math
import time

# Utility function to use cv.imshow in jupyter notebook (lazy to type waitKey and destroyAllWindows)
def show_img(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Config models path
face_proto = "Age and Gender Detection Model/opencv_face_detector.pbtxt"
face_model = "Age and Gender Detection Model/opencv_face_detector_uint8.pb"

age_proto = "Age and Gender Detection Model/age_deploy.prototxt"
age_model = "Age and Gender Detection Model/age_net.caffemodel"

gender_proto = "Age and Gender Detection Model/gender_deploy.prototxt"
gender_model = "Age and Gender Detection Model/gender_net.caffemodel"

# Config static values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load model
age_net = cv.dnn.readNet(age_model, age_proto)
gender_net = cv.dnn.readNet(gender_model, gender_proto)
face_net = cv.dnn.readNet(face_model, face_proto)


# Function to return image with face box and the position of the box
def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    blob = cv.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])
            cv.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    return frame_opencv_dnn, boxes


# Function to retun image with prediction of age and gender
def gender_and_age_detection(frame):
    frame_face, boxes = get_face_box(face_net, frame)
    padding = 20
    for box in boxes:
        face = frame[max(0, box[1] - padding):min(box[3] + padding, frame.shape[0] - 1),
               max(0, box[0] - padding):min(box[2] + padding, frame.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        gender_net.setInput(blob)
        gender_predicted = gender_net.forward()
        gender = gender_list[gender_predicted[0].argmax()]

        age_net.setInput(blob)
        age_predicted = age_net.forward()
        age = age_list[age_predicted[0].argmax()]

        label = '{},{}'.format(gender, age)
        cv.putText(frame_face, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
    return frame_face


#live
live = cv.VideoCapture(0)
while True:
    ret, frame = live.read()
    cv.imshow('Gender and Age Detector',gender_and_age_detection(frame))
    check = cv.waitKey(1)
    if check == ord('q'):
        break

live.release()
cv.destroyAllWindows()




