# Surveillance-System
A pipeline that acquires image from multiple CCTV cameras and carry out face detection, face recognition and tracking of selected individuals. 
* Acquisition :Multiple static CCTV cameras are considered.
* Face detection & Recognition: detect the faces and recognize the individuals 
* Multiple Person Tracking: Out of the recognized individuals, track target individuals across multiple cameras

Some points to note - 
* Each persons unique face encoding is captured and stored in database and unique ID assigned when person when he/she is captured for first time by system
* Each captured face encoding is matched with already stored encodings and if a match is found it means that the person was captured before and time/location fields are updated 
* Persons location and time is stored in database
