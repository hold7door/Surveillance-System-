import face_recognition
import cv2
from objecttracker.centroidtracker import CentroidTracker
import json
import MySQLdb

db = MySQLdb.connect(user = "", passwd = "", db = "")
c = db.cursor()

video_capture = cv2.VideoCapture(0)

ct = CentroidTracker(db, c)
                                                                                        # Initialize some variables
face_locations = []

while True:
                                                                                        # Grab a single frame of video
    ret, frame = video_capture.read()
                                                                                            # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                                                                                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
                                                                                        # Only process every other frame of video to save time
    if process_this_frame:
                                                                                        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    boxes = []
                                                                                        # Display the results
    for (top, right, bottom, left) in face_locations:
                                                                                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        boxes.append((left, top, right, bottom ))
                                                                                        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
    object_coord, object_en = ct.update(boxes, face_encodings)
    for objectID, centroid in object_coord.items():
        text = "ID {}".format(objectID)
        en = object_en[objectID]
        j_en = json.dumps(en.tolist())                                                  #Convert ndarray to list and then json serialize to store into database
        try:                                                                                
            c.execute("""Insert into person values(%s, %s)""", (objectID, 1))               #Second argument should be unique identifier to a camera
            print "[*] New identity inserted into database"
        except:                                                                             #Occurs if objectID already exists in database
            print "[*] Identity already exists in database. Adding face encodings'"
        c.execute(""" Insert into face_encodings values (%s, %s)""", (objectID, j_en))
        db.commit()
        cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
                                                                                        # Display the resulting image
    cv2.imshow('Video', frame)
                                                                                        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
                                                                                    # Release handle to the webcam
cv2.destroyAllWindows()