import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


# Init
face_cascade_name = 'data/haarcascade_frontalface_default.xml'
eyes_cascade_name = 'data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Loading the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('Error loading eyes cascade')
    exit(0)

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]
        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv2.imshow('Capture - Face detection', frame)

def normal_stream():
    cap = cv2.VideoCapture(0)  
    frames = 0
    # Start time measurement
    start = time.time()
    while(True):
        ret, frame = cap.read()        
        frames += 1
        # Display the frames
        cv2.imshow('frame',frame)
        # End time measurement
        end = time.time()
        elapsed_time = end - start
        # Wait for a second and print the number of frames 
        if elapsed_time >= 1:
            print (frames, "fps")
            frames = 0
            start = end
        
        # Exiting the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def good_features(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    #plt.imshow(img),plt.show()
    #cv2.imshow('Capture - Face detection', img)
    return img
    
def face_detection_stream():
    cap = cv2.VideoCapture(0)
    frames = 0
    # Start the time measurement
    start = time.time()
    while(True):
        ret, frame = cap.read()
        frames += 1
        # Run the cascade face detector
        frame = good_features(frame)
        detectAndDisplay(frame)
        # End the time measurement
        end = time.time()
        elapsed_time = end - start
        # Wait for a second and print the number of frames 
        if elapsed_time >= 1:
            print (frames, "fps")
            frames = 0
            start = end

        # Press Q to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
normal_stream()
face_detection_stream()