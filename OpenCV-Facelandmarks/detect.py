# import the opencv library
from email import header
import math
import cv2, dlib
from facePoints import facePoints
import os , subprocess
from typing import Union
# define a video capture object
vid = cv2.VideoCapture(0)


def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
    with open(fileName, 'w') as f:
        for p in faceLandmarks.parts():
            f.write("%s %s\n" % (int(p.x), int(p.y)))

    f.close()

def sum(p0, p1):
    return math.sqrt((p0.x-p1.x)**2+(p0.y-p1.y)**2)


firefoxPath = "C:\Program Files\Mozilla Firefox\firefox.exe"
def angle(p0, p1):
    return math.atan((p0.x-p1.x)/(p0.y-p1.y))

firefoxPath = "C:\Program Files\Mozilla Firefox\firefox.exe"

def open_firefox(url: Union[str, None]):
    try:
        os.system(firefoxPath)
        subprocess.Popen([firefoxPath] + url)
    except:
        print("File might not be found, due to software-restriction")
    print("Firefox has been opened because of the users reaction!")


def head_inclined_reaction():
    open_firefox("https://stackoverflow.com/")
    print("Opened StackOverflow because of the users reaction!")


def eye_lids_up_reaction():
    open_firefox("https://github.com/Ventrum")
    print("Opened Github because of the users reaction!")

def analyze(faceLandmarkDetector, xy):
    # differenz first to last jaw point
    jaw_difference = sum(faceLandmarkDetector.part(0),faceLandmarkDetector.part(16))
    left_eyelid_difference = (sum(faceLandmarkDetector.part(23), faceLandmarkDetector.part(47))+ sum(faceLandmarkDetector.part(24), faceLandmarkDetector.part(46)))/2
    right_eyelid_difference = (sum(faceLandmarkDetector.part(19), faceLandmarkDetector.part(41))+ sum(faceLandmarkDetector.part(20), faceLandmarkDetector.part(40)))/2
    left_eye_difference = (sum(faceLandmarkDetector.part(43), faceLandmarkDetector.part(47))+ sum(faceLandmarkDetector.part(44), faceLandmarkDetector.part(46)))/2
    right_eye_difference = (sum(faceLandmarkDetector.part(37), faceLandmarkDetector.part(41))+ sum(faceLandmarkDetector.part(38), faceLandmarkDetector.part(40)))/2
    head_angle = angle(faceLandmarkDetector.part(27),faceLandmarkDetector.part(8))/math.pi*360
    mouth_inner_height = (sum(faceLandmarkDetector.part(62), faceLandmarkDetector.part(66)))
    print("-")
    if False:
        print("eyes winked")
    if right_eyelid_difference/ jaw_difference > 0.25 or left_eyelid_difference/ jaw_difference > 0.25:
        print("eye-lids-up")
        eye_lids_up_reaction()
    if  left_eye_difference/ jaw_difference < 0.035 or right_eye_difference/ jaw_difference < 0.035:
        print("Eyes closed")
        #open_firefox()
    if False:
        print("mouth-courner up")
    if False:
        print("mouth-courner down")
    if mouth_inner_height/jaw_difference > 0.05:
        print("mouth opend")
    if head_angle < -30 or 30 < head_angle :
        print("head inclined")
        head_inclined_reaction()

    

# location of the model (path of the model).
Model_PATH = "shape_predictor_68_face_landmarks.dat"

# now from the dlib we are extracting the method get_frontal_face_detector()
# and assign that object result to frontalFaceDetector to detect face from the image with
# the help of the 68_face_landmarks.dat model
frontalFaceDetector = dlib.get_frontal_face_detector()

# Now the dlip shape_predictor class will take model and with the help of that, it will show
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# List to store landmarks of all detected faces
allFacesLandmark = []


while (True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    allFaces = frontalFaceDetector(imageRGB, 0)

    xy = 0
    for k in range(0, len(allFaces)):
        # dlib rectangle class will detecting face so that landmark can apply inside of that area
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                           int(allFaces[k].right()), int(allFaces[k].bottom()))

        # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

        # Analyziation
        #print(detectedLandmarks.part(49) - detectedLandmarks.part(62))
        analyze(detectedLandmarks, xy)

        # Svaing the landmark one by one to the output folder
        allFacesLandmark.append(detectedLandmarks)

        # Now finally we drawing landmarks on face
        facePoints(frame, detectedLandmarks)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
