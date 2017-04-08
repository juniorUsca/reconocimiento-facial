import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

###source = "rtsp://admin:12345@20.0.0.14//Streaming/Channels/2"
source = "rtsp://admin:Hik12345@172.16.1.52:554/h264/ch16/main/av_stream"
cap = cv2.VideoCapture(0)

ok_flag = True
while ok_flag:
    (ok_flag, img) = cap.read()
    if not ok_flag: break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2, # redimensiona la imagen, a mayor escala mas rapido y mayor perdida
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("some window", img)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()


