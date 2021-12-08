from collections import OrderedDict
import time

class FCWAlert:
    def __init__(self):
        self.objects = OrderedDict()
        self.previousTime = time.time()
        self.frame_counter = 0
        self.prev_dist = 0
        self.alert_counter = 0
        
    def getSpeed(self, prevDistance, currentDistance, time_diff):       
        scale_change = prevDistance / currentDistance
        result = time_diff/(scale_change - 1)
        #print("prevDistance: ", prevDistance)  # in meter
        #print("currDistance: ", currentDistance)  # in meter
        #print("TTCresult: ", result) # in meter
        #print("time_diff: ", time_diff) # in meter/sec
        return result  
        
    def calculateObjectSpeeds(self, objects):
        objectsWithSpeed = OrderedDict()
        if len(self.objects) == 0:
            self.objects = objects
            return None
        #print('')
        currentTime = time.time()
        timeDifference = currentTime - self.previousTime
        #print('Current Time : ', currentTime)
        #print('Previous Time : ', self.previousTime)
        #print("Time Difference", timeDifference) # in sec
        self.previousTime = currentTime
        #print('objectPreviousData in calculateObjectSpeed ::: ', self.objects)
        
        for objectID, (cX, cY, startX, startY, endX, endY, distance) in objects.items():
            
            #print('objectID : ', objectID)
            objectPreviousData = self.objects.get(objectID, None)
            #print('objectPreviousData ::: ', self.objects)
            
            if objectPreviousData is None:
                self.objects[objectID] = (cX, cY, startX, startY, endX, endY, distance)
                continue #theres no matching value
                
            #print('distance :: ', distance) 
            prevDistance = objectPreviousData[6]
            #print('prevDistance :: ', prevDistance)
            result = self.getSpeed(prevDistance, distance, timeDifference)
            objectsWithSpeed[objectID] = (cX, cY, startX, startY, endX, endY, distance, result)
            #print('objects : ', objects)\

        self.objects = objects.copy()            
        #print('objectPreviousData ::: ', self.objects)
        
        return objectsWithSpeed
    
    def checkFcwAlert(self, distance, result):

        #TTC = distance/relative_velocity
        #print('ObjectID :', objectID)
        #print('Fcw Distance : ', distance)
        #print('Relative_velocity : ', relative_velocity)
        #print('TTC Value : ', result)
        #print('')
        if 0.0 < result < 2.3 :
            #print('FCWS Alert')
            #print('')
            return True
                    
        else:
            return False
        
    def checkSfdAlert(self, distance, vehicle_speed):

        speed_meter_per_sec = float(vehicle_speed)*0.2778 
        safe_following_dist = distance / speed_meter_per_sec
        
        #print('distance : ', distance)
        #print('safe_following_dist : ', safe_following_dist)
        
        if safe_following_dist < 1.0:
            return True, safe_following_dist

        else:
            return False, safe_following_dist
                
            
    def update(self, objects):
        if objects is None:
            return None
        
        objectsWithSpeed = self.calculateObjectSpeeds(objects)
        #print("objects with speed: ", objectsWithSpeed)
        return objectsWithSpeed
    
    def checkMtAlert(self, objectsWithSpeed):
    
        #print('self.alert_counter', self.alert_counter)
        '''
        if self.alert_counter < 4:
            self.alert_counter += 1
            return False
        '''
        if objectsWithSpeed is not None:
            traffic_object = objectsWithSpeed.copy()               
            frontVehicle = traffic_object.popitem()
                
            if self.prev_dist == 0:
                #print('Getting prev distance')
                self.prev_dist = frontVehicle[1][6] 
                return False                  
                
            else:
                curr_dist = frontVehicle[1][6]
                dist_diff = curr_dist - self.prev_dist
                #print('')
                #print('self.prev_dist : ', self.prev_dist)
                #print('dist_diff : ', dist_diff)
                
                if dist_diff > 1.0:
                    # alert on
                    return True
            
                else:
                    # alert off
                    return False
        else:
            return False

    def resetMtAlert(self):
        self.prev_dist = 0
        self.alert_counter = 0
        
                
            
            
        
        
