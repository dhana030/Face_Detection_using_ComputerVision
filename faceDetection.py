import cv2

alg = "haarcascade_frontalface_default.xml" #declaring the file
haar_cascade = cv2.CascadeClassifier(alg)   #looading the file

cam = cv2.VideoCapture(0) #intializing the camera

while True:
    _,img = cam.read() #reading the frame

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #preprocessing should be done as done in the model.

    face = haar_cascade.detectMultiScale(grayImg,1.3,4) # using this function detecting the face features and getting the coordinates

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Face Detection ",img)
    key = cv2.waitKey(10)
    if key == 0:
        break
cam.release()
cv2.destroyAllWindows()


