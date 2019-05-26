from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import json
import face_recognition

class CentroidTracker:
    def __init__(self, db, c, maxDisappeared=500):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_en = OrderedDict()
        self.db = db                    
        self.c = c                  #database cursor
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid, encoding):
        self.objects[self.nextObjectID] = centroid
        self.objects_en[self.nextObjectID] = encoding
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID +=1
    
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects, face_encod):
        if not len(rects):                                                          #No face found in frame. Disappeared value of all faces is incremented by 1
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] +=1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            return self.objects, self.objects_en
        
        inputCentroids = np.zeros((len(rects), 2) , dtype='int')
        
        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX)/2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            
        if not len(self.objects):                                                   #No face exists. Register all new faces
            for i in range(len(rects)):
                self.register(inputCentroids[i], face_encod[i])
            
        else:
            objectIDs = list(self.objects.keys())                                   #Existing face ids
            objectCentroids = self.objects.values()                                 #Existing Faces
            
            
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            #print len(objectCentroids), len(inputCentroids), D
            rows = D.min(axis=1).argsort()
                                                # WHAT ??!!
            cols = D.argmin(axis=1)[rows]
            
            usedrows = set()
            usedcols = set()
            
            for (row, col) in zip(rows, cols):
                if row in usedrows or col in usedcols:
                    continue
                    
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.objects_en[objectID] = face_encod[col]
                self.disappeared[objectID] = 0
                
                usedrows.add(row)
                usedcols.add(col)
                
            unusedrows = set(range(D.shape[0])).difference(usedrows)
            unusedcols = set(range(D.shape[1])).difference(usedcols)
            
            if D.shape[0] >= D.shape[1]:
                                                                                                #Number of new faces is less than existing faces.
                for row in unusedrows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] +=1
                    
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                    
            else:
                print "[*] New Faces Found. Checking if the Unknown Face is present in database"
                self.c.execute("""Select id, encoding from face_encodings""")
                res = self.c.fetchall()
                for col in unusedcols:
                    unknown_face_encoding = face_encod[col]
                    id = ""
                    for id, encod in res:
                        known_face_encoding = [np.array(json.loads(encod))]
                        dis = face_recognition.face_distance(known_face_encoding, unknown_face_encoding)
                        match = False
                        if dis[0] < 0.30:                                                       #Sample Threshold value
                            print "[*] Face matched in database with id {0}".format(id)
                            match = True
                            id = int(id)
                            break
                    if match:
                        print "[*] Assigning known Id"
                        self.objects[id] = inputCentroids[col]
                    else:   
                        print "[*] Face not found. Registering and assigning new Id"
                        self.register(inputCentroids[col], face_encod[col])
            
        return self.objects, self.objects_en
            
            
            