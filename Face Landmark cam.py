import cv2
import numpy as np
import dlib,time

cap= cv2.VideoCapture(0)
time.sleep(2)

# Create the haar cascade
faceCascade =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create the landmark predictor
predictor =  dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while True:

    ret, image = cap.read()
    if ret:
        image = cv2.flip(image, 1)
        # image = cv2.resize(image, (400,400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = predictor(image, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            # print landmarks

            for idx,point in enumerate(landmarks):
                pos=(point[0,0],point[0,1])

                cv2.circle(image,pos,2,color=(0,0,255),thickness=-5)

        cv2.imshow('Landmark found',image)

    k = cv2.waitKey(30) & 0xff

    # if the 'q' key is pressed, stop the loop
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
