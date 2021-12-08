import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class DetectedObjects:
    def __init__(self,maxDisappeared=5,pixelToMeterRatio=104.5,startX=100,endX=200):
        self.centroidTracker = CentroidTracker(maxDisappeared)
        self.objectDistanceCaculator = ObjectDistances(pixelToMeterRatio)
        self.objectFilter=ObjectFilter(startX,endX)
        
    def update(self, rects):
        #first get the centroids
        objects = self.centroidTracker.validate_object(rects)
        objects = self.centroidTracker.update(rects)
        #print("objects: ", objects)
        if objects is None:
            return None
            
        #then filter the objects that aren't in view
        filteredObjects = self.objectFilter.determineObjectsInFrontOfVehicle(objects)
        #print("Filtered Objects: ",filteredObjects)
        if filteredObjects is None:
            return None
            
        #then find out the distances
        objectsWithDistance = self.objectDistanceCaculator.calculateDistances(filteredObjects)
        #print("Objects with distance: ", objectsWithDistance)
        return objectsWithDistance
       
class ObjectFilter:
    def __init__(self,startX=100,endX=200):
        self.startX = startX
        self.endX = endX
    
    def determineObjectsInFrontOfVehicle(self, objects):
        filteredObjects = OrderedDict()
        if len(objects) == 0:
            return None

        else: 
            for key, (cX, cY, startX, startY, endX, endY) in objects.items():
                if cX < self.startX or cX > self.endX:
                    continue
                else:
                    data = (cX, cY, startX, startY, endX, endY)
                    filteredObjects[key] = data
                    
            return filteredObjects
    
class ObjectDistances:
    def __init__(self,pixelToMeterRatio=104.5):
        self.pixelToMeterRatio = pixelToMeterRatio #e.g. 100 pixel = 1meter
        # self.widthOfCar =  
    
    def calculateDistanceFromWidth(self,width):
        return self.pixelToMeterRatio/width #e.g. (100 pixels/x num of pixels) * 1m/pixel = real meters        
        
    def calculateDistances(self, objects):
        allDistances = OrderedDict()
        
        objectIDs = list(objects.keys())
        objectCentroids = list(objects.values())
        
        if len(objects) == 0:
            return None
        
        else: 
            for key, (cX, cY, startX, startY, endX, endY) in objects.items():
                width = endX-startX
                distance = (self.calculateDistanceFromWidth(width))*10
                data = (cX, cY, startX, startY, endX, endY, distance)
                allDistances[key] = data
            return allDistances

class CentroidTracker:
    def __init__(self, maxDisappeared=5):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.appeared = OrderedDict()
        self.disappeared = OrderedDict()
        self.tempObjects = OrderedDict()
        
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def validate_object(self, rects):
        #print('updating')
        #print('appear :: ', self.appeared)
        #print('disappeared :: ', self.disappeared)
        
        for objectID in list(self.appeared.keys()):
            self.appeared[objectID] += 1
        
            if self.appeared[objectID] == 2:
                #print('return objects')
                self.objects[objectID] = self.tempObjects[objectID]
        
        return self.objects
        
    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        #print('register')
        self.tempObjects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
        if self.nextObjectID > 1000:
            self.nextObjectID = 0

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects):
        #print('UPDATING OBJECT')
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects
            
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 6), dtype="int")
        # loop over the bounding box rectangles
        #print('debug 1')
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY, startX, startY, endX, endY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            #print('debug 2')
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

