import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import multiprocessing as mp
import coco
import vehicle
import sys
import serial
import threading

# import uff
import tensorrt as trt
#import graphsurgeon as gs
#from config import model_ssd_inception_v2_coco_2017_11_17 as model
#from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
#from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model
from config import delloyd_lane as model
from final_ldw_helper_functions import *
from final_lane_detector_functions import * 

from object_tracker import DetectedObjects
from fcw_alert import FCWAlert

detection_file_path = './detection_output'
detection_image = []

dewarp_mask = np.zeros((340,620,1), np.uint8)   # VIDEO
detection_mask = np.zeros((340,620,1), np.uint8)
fcw_counter = 0
MtAlert_Counter = 0
hms_counter = 0
prev_velocity = '0'

inFilterLeft = 295
inFilterRight = 340
mta_set_flag = False

'''
port = "/dev/ttyACM0"
ser = serial.Serial(port, baudrate = 9600, timeout = None)
'''
output_1 = cv2.VideoWriter('live_feed.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 12, (1280,720))
output_2 = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 8, (620,340))
output_3 = cv2.VideoWriter('fcw-result.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 8, (620,340))

def grab_frame(objectDetectorBuffer,laneImageBuffer,liveViewBuffer):

    cap = cv2.VideoCapture('/home/petronas/repos/record_vid/malam-highway2.avi')
    #cap = cv2.VideoCapture(0)
    #average_time = [] #FPS 
    #frame = cv2.imread('/home/petronas/repos/snap_from_video/1_meter.jpg')
    ret,frame=cap.read()
    #cv2.imshow("ADAS DISPLAY", frame)
    #cv2.moveWindow("ADAS DISPLAY", 10,10)
    
    while True:
        #start_time = time.time()
        ret,frame=cap.read()
        #frame = cv2.imread('/home/petronas/repos/image_for_calib/malam2.png')
        #output_1.write(frame)
        
        #frame = scale_to_fixed(frame)
        #(h2, w2) = frame.shape[:2]
        frame = cv2.resize(frame, (640, 360))  # malam-hujan
        (h2, w2) = frame.shape[:2]
        
        cropYTop = 10
        cropYBot = 10
        cropXLeft = 10
        cropXRight = 10

        frame = frame[cropYTop:h2-cropYBot,cropXLeft:w2-cropXRight]
        laneImageBuffer.put(frame)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(0)

        objectDetectorBuffer.put(frame)
        #current_time = 1/(time.time()-start_time)  #FPS
        
        liveViewBuffer.put(frame)
                
        #average_time.append(current_time) #FPS
        #print("fps "+str(np.mean(average_time))) #FPS
        #cv2.waitKey(0)
        
    cap.release()

def lane_detector(queue, lane_result, mask_img):
    frame = queue.get(True)

    read_calibration_value()
    
    while True:
        #start_time = time.time() 
        frame = queue.get(True)
        lane_value = run_pipeline(frame, mask_img)
        lane_result.put(lane_value)
        #print("fps "+str(1/(time.time()-start_time)))
        #print('Lane execution time : ', str(time.time()-start_time))
        cv2.waitKey(1)
        
def obj_detection_fx1(fcwAlert, vehicle_speed, distance, result, objectsWithSpeed):
    global fcw_counter, MtAlert_Counter, mta_set_flag
    if float(vehicle_speed) > 70 :
        if fcwAlert.checkFcwAlert(distance, result):
            #print('')
            #print('OBJECT ID: ', objectID)
            fcw_counter = 4
            #cv2.putText(frame, "FCWS Alert!!", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Moving Traffic Alert      # TO DO: diff trigger
    elif float(vehicle_speed) < 2 :   #check speed at traffic light and starts to move
        if fcwAlert.checkMtAlert(objectsWithSpeed):
            #cv2.putText(frame, "Moving Traffic Ahead!!", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            MtAlert_Counter = 5
            mta_set_flag = True


def obj_detection_fx2(fcwAlert, vehicle_speed, distance, hms_dist_list):
    global hms_counter

    # Headway Monitoring System     # TO DO: diff trigger
    if float(vehicle_speed) > 40 :
        hms_result, following_dist = fcwAlert.checkSfdAlert(distance, vehicle_speed)
        hms_dist_list.append(following_dist)
        if hms_result is True:
            #print('set flag')
            hms_counter = 3
            #cv2.putText(frame, "Vehicle Distance Alert!!", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def run_detection(queue,object_detector_result):
    global fcw_counter, MtAlert_Counter, hms_counter, inFilterLeft, inFilterRight, prev_velocity, mta_set_flag
    
    is_fcw_triggered = False
    is_mta_triggered = False
    is_hms_triggered = False
    obj_result = []
    #cap = cv2.VideoCapture("./dashcam/EVT1_20171207_045703.avi")    
    ctypes.CDLL("lib/libflattenconcat.so")
    #COCO_LABELS = coco.COCO_CLASSES_LIST
    COCO_LABELS = vehicle.VEHICLE_CLASSES_LIST 
    #initialize
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)

    print("create engine")
    # create engine
    with open(model.TRTbin, 'rb') as f:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)
        
    # create buffer
    host_inputs  = []
    cuda_inputs  = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()
    
    #Object tracker
    detectedObjects = DetectedObjects(maxDisappeared=2,pixelToMeterRatio=104.5,startX=255,endX=380)
    fcwAlert = FCWAlert()
    
    while True:
    
        '''
        data = ser.readline()
        data_decode = data.decode('latin-1')
        vehicle_speed = parseGPS_velocity(data_decode)
        '''
        vehicle_speed = '80'
    
        rects =[]
        hms_dist_list = []
        #start_time = time.time() #fps
        
        #ret, frame = cap.read() # true means block until queue is not empty
        frame = queue.get(True)        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (model.dims[2],model.dims[1]))
        image = (2.0/255.0) * image - 1.0
        image = image.transpose((2, 0, 1))
        
        np.copyto(host_inputs[0], image.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        
        output = host_outputs[0]
        #print("put output detection")
        
        height, width, channels = frame.shape
        for i in range(int(len(output)/model.layout)):
            prefix = i*model.layout
            index = int(output[prefix+0])
            label = int(output[prefix+1])
            
            conf  = output[prefix+2]
            xmin  = int(output[prefix+3]*width)
            ymin  = int(output[prefix+4]*height)
            xmax  = int(output[prefix+5]*width)
            ymax  = int(output[prefix+6]*height)
            
            if conf > 0.5:
                rects.append(np.array([xmin,ymin,xmax,ymax]))
                #print(label)
                #print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
                #cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (255,161,107),2)
                #cv2.putText(frame, test_text,(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                
        objects = detectedObjects.update(rects)
        objectsWithSpeed = fcwAlert.update(objects)
        '''
        leftFilterThresh = detectedObjects.objectFilter.startX
        rightFilterThresh = detectedObjects.objectFilter.endX
        cv2.line(frame,(leftFilterThresh,0),(leftFilterThresh,height),(0,255,0),2)
        cv2.line(frame,(rightFilterThresh,0),(rightFilterThresh,height),(0,255,0),2)
        
        cv2.line(frame,(inFilterLeft,0),(inFilterLeft,height),(240,0,0),2)
        cv2.line(frame,(inFilterRight,0),(inFilterRight,height),(240,0,0),2)
        '''

        if vehicle_speed is None or vehicle_speed == '':
            vehicle_speed = prev_velocity
        else:
            prev_velocity = vehicle_speed
        
        ## START
        
        if objectsWithSpeed is not None:           
            for objectID, (cX, cY, startX, startY, endX, endY, distance,result) in objectsWithSpeed.items():
                
                # if((near OR far) AND distance_limit AND (width < 300)    
                if (distance < 22 or (cX > inFilterLeft and cX < inFilterRight)) and (distance < 35) and ((endX - startX) < 400): 
                    
                    # FCWS
                    #dist_in_meter = distance
                    #text = "ID {}, Distance: {:.2f}".format(objectID, dist_in_meter)
                    #cv2.putText(frame, text, (cX - 10, cY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)
                    #cv2.rectangle(frame,(startX,startY), (endX,endY), (0,230,255), 3)
                    thread_1 = threading.Thread(target = obj_detection_fx1, args = (fcwAlert, vehicle_speed, distance, result, objectsWithSpeed))
                    thread_2 = threading.Thread(target = obj_detection_fx2, args = (fcwAlert, vehicle_speed, distance, hms_dist_list))
                    thread_1.start()
                    thread_2.start()
                    thread_1.join()
                    thread_2.join()
                    
                else:
                     continue
                        
        
        ## fcw counter
        if fcw_counter > 0:
            fcw_counter -= 1
        
        if fcw_counter == 0 :
            is_fcw_triggered = False
        else:
            is_fcw_triggered = True
            
        ## mta counter    
        if MtAlert_Counter > 0:
            MtAlert_Counter -= 1
        
        if MtAlert_Counter == 0 :
            is_mta_triggered = False
            
            if mta_set_flag is True:
                fcwAlert.resetMtAlert()
                hms_dist_list.clear()
                mta_set_flag = False
            
        else:
            is_mta_triggered = True
            
        ## hms counter  
        if hms_counter > 0:
            hms_counter -= 1
        
        if hms_counter == 0 :
            is_hms_triggered = False
        else:
            is_hms_triggered = True

        ### final detection result
        if is_fcw_triggered is True or is_mta_triggered is True or is_hms_triggered is True:
            obj_detection = 1
        else:
            obj_detection = 0 

        ## END           
        #output_3.write(frame)
        if len(hms_dist_list) != 0:
            hms_dist = min(hms_dist_list)
            if hms_dist > 2.0:
                hms_dist = 2.0
        else: 
            hms_dist = 2.0
            
        obj_result.insert(0, obj_detection)
        obj_result.insert(1, hms_dist)
        obj_result.insert(2, vehicle_speed)

        object_detector_result.put(obj_result)    # send detection trigger as number

        #print("fps "+str(1/(time.time()-start_time)))
        #cv2.imshow("result", frame)
        cv2.waitKey(1)
        #print("execute times "+str(time.time()-start_time))

def post_processing(object_detector_result, lane_result, queue):
    global detection_image
    
    while True:
        obj_result = object_detector_result.get(True)        
        ldw_detection_value = lane_result.get(True)
        live_frame = queue.get(True)
        obj_detection_value = obj_result[0]
        hms_dist = obj_result[1]
        vehicle_speed = obj_result[2]
        
        if obj_detection_value == 1 and ldw_detection_value == 0:
            adas_alert = detection_image[0]
            
        elif obj_detection_value == 1 and ldw_detection_value == 1:
            adas_alert = detection_image[1]

        elif obj_detection_value == 1 and ldw_detection_value == 2:
            adas_alert = detection_image[2]
            
        elif obj_detection_value == 0 and ldw_detection_value == 0:
            adas_alert = detection_image[3]
            
        elif obj_detection_value == 0 and ldw_detection_value == 1:
            adas_alert = detection_image[4]
            
        elif obj_detection_value == 0 and ldw_detection_value == 2:
            adas_alert = detection_image[5]
        
        adas_alert = cv2.resize(adas_alert,(int(adas_alert.shape[1]/4), int(adas_alert.shape[0]/4)))
        #cv2.imshow('ALERT', adas_alert)
        x_offset = 450
        y_offset = 10
        live_frame[y_offset:y_offset+adas_alert.shape[0], x_offset:x_offset+adas_alert.shape[1]] = adas_alert

        if hms_dist < 1.0:
            color = (0, 0, 220)
        else:
            color = (0, 240, 0)

        hms_text = "{:.1f}".format(hms_dist)
        speed_text = "Speed : {:.1f}".format(float(vehicle_speed))
        cv2.putText(live_frame, hms_text, (live_frame.shape[1]-106, live_frame.shape[0]-230),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(live_frame, speed_text, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        cv2.imshow('ADAS OUTPUT', live_frame)
        #output_2.write(live_frame)
        cv2.waitKey(1)   

def initProgram():
    global detection_file_path, detection_image
    
    file_list = os.listdir(detection_file_path)
    
    for filename in sorted(file_list):
        img_temp = cv2.imread(os.path.join(detection_file_path + '/' + filename))
        detection_image.append(img_temp)

def parseGPS_velocity(data):

    req_data = data.find('$GPVTG')
    
    if req_data != -1:
        #print('--Receiving Velocity Data....')
        sdata = data.split(",")
        
        speed_value = sdata[7]
        #print('Speed : ', speed_value)
        
        return speed_value

def grab_ldw_mask_img():
    global dewarp_mask
    ldw_mask_min = 150
    ldw_mask_max = 470
     
    for x in range(ldw_mask_min, ldw_mask_max):
        for y in range(0, dewarp_mask.shape[0]):
            dewarp_mask[y][x] = [255]
            
    return dewarp_mask
    
if __name__ == '__main__':
        
    initProgram()
    mask_img = grab_ldw_mask_img()
    
    imageBuffer = mp.Queue(1)
    laneImageBuffer = mp.Queue(1)
    liveViewBuffer = mp.Queue(1)
    
    object_detector_result = mp.Queue(1)
    lane_result = mp.Queue(1)
    print("start process")
    grabber = mp.Process(target=grab_frame,args=(imageBuffer,laneImageBuffer, liveViewBuffer))
    grabber.daemon = True
    grabber.start()
    
    object_detector = mp.Process(target=run_detection, args=(imageBuffer,object_detector_result))
    object_detector.daemon = True
    object_detector.start()
    
    lane_detector = mp.Process(target=lane_detector, args=(laneImageBuffer,lane_result,mask_img))
    lane_detector.daemon = True
    lane_detector.start()
    
    post_process = mp.Process(target=post_processing, args=(object_detector_result,lane_result,liveViewBuffer))
    post_process.daemon = True
    post_process.start()
    
    while True:
        pass
    cv2.waitKey(0)
    
