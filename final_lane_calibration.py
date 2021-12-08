from final_ldw_helper_functions import *
import cv2
import numpy as np
import os.path
countL = 0
countR = 0
c = 0
number = 0
counter = 0

ref_left = np.array([-0.0001, 0, 300])
ref_right = np.array([-0.0001, 0, 500])
left_fit = np.array([-0.0001, 0, 300])
right_fit = np.array([-0.0001, 0, 500])

currentL = np.zeros([3])
previousL = np.zeros([3])
currentR = np.zeros([3])
previousR = np.zeros([3])

DAY_BRIGHTNESS_TRESH = 30
#dewarp_mask = np.zeros((240,600,1), np.uint8)
dewarp_mask = np.zeros((300,620,1), np.uint8)   # VIDEO
mask_min = 150
mask_max = 470

image_counter = 0
save_path = '/home/petronas/repos/snap_from_video/debugging/'

detection_file_path = './detection_output'
detection_image = []

def main():
    global detection_file_path, detection_image
    
    file_list = os.listdir(detection_file_path)
    
    for filename in sorted(file_list):
        img_temp = cv2.imread(os.path.join(detection_file_path + '/' + filename))
        detection_image.append(img_temp)
    
    cap = cv2.VideoCapture('/home/petronas/repos/record_vid/siang1.avi')
    ret,frame=cap.read()
    #frame = cv2.imread('/home/petronas/repos/snap_from_video/debugging/frame31.jpg')
    (h, w) = frame.shape[:2] 
    read_calibration_value()                                         
    createtrackbar(h, w)

    for x in range(mask_min, mask_max):
        for y in range(0, dewarp_mask.shape[0]):
            dewarp_mask[y][x] = [255]

    while True:
        ret,frame=cap.read()
        #frame = cv2.imread('/home/petronas/repos/snap_from_video/debugging/frame31.jpg')
        frame = scale_to_fixed(frame)
        (h2, w2) = frame.shape[:2]
        
        cropYTop = 10
        cropYBot = 50
        cropXLeft = 10
        cropXRight = 10
    
        frame = frame[cropYTop:h2-cropYBot,cropXLeft:w2-cropXRight]
        run_pipeline(frame)
        cv2.waitKey(10)
        
    cap.release()

def run_pipeline(frame):
    global ref_left, ref_right, image_counter, save_path, detection_file_path, detection_image

    w_Key = cv2.waitKey(5)
    
    if w_Key%256 == 32:
        
        image_counter = image_counter + 1
        print('Saving File : ', image_counter)
        img_name = + "frame_" + str(image_counter) + '.jpg'
        filename = os.path.join(save_path, img_name) 
        cv2.imwrite(filename, frame)
    
    cv2.imshow('FRAME', frame)

    # Pre-processing
    

    """STEP#0 SCALE TO FIXED"""
    """
    frame = scale_to_fixed(frame)
    (h2, w2) = frame.shape[:2]
    
    cropYTop = 10
    cropYBot = 50
    cropXLeft = 10
    cropXRight = 10
    
    frame = frame[cropYTop:h2-cropYBot,cropXLeft:w2-cropXRight]

    brightness = getBrightness(frame)
    """

    # ---------LDW PIPELINE
    """LDW-STEP#1 CROP ROAD ROI"""
    #crop_road_roi_frame = crop_road_roi(frame, frame_output=False)
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    crop_road_roi_frame = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel_1)
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    crop_road_roi_frame = cv2.erode(crop_road_roi_frame, kernel_1)
    # ori_color = apply_perspective_transform(frame.copy(), crop_road_roi_frame, toWarp=True)
    dewarped_color = apply_perspective_transform(frame.copy(), frame.copy(), toWarp=True)
    cv2.imshow("dewarped_color", dewarped_color)
    crop_road_roi_frame = cv2.Canny(crop_road_roi_frame, 100, 200)

    """LDW-STEP#2 PERSPECTIVE TRANSFORM"""
    warped_image = apply_perspective_transform(frame, crop_road_roi_frame, toWarp=True)
    #print('warped image shape', warped_image.shape)
    #print('warped image dtype', warped_image.dtype)
    #print('mask image shape', dewarp_mask.shape)
    #print('mask image dtype', dewarp_mask.dtype)
    warped_image_masked = cv2.bitwise_and(warped_image, dewarp_mask)
    cv2.imshow('dewarped_binary', warped_image_masked)
    blur = cv2.GaussianBlur(warped_image_masked, (5, 5), 0)

    """LDW-STEP#3 TIME CONDITION CHECK & BINARIZE"""
    sharpened_image = sharpened(blur)
    binaryImage = cv2.Canny(warped_image_masked,100,200)
    whites = cv2.countNonZero(binaryImage)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel_2)

    """LDW-STEP#4 DEFINE LANE-LINES"""
    #cv2.imshow('DISPLAY', binaryImage)
    #cv2.waitKey(0)
    leftx, lefty, rightx, righty, found, window_detection = extract_lanes_pixels(binaryImage, plot_show=False)
    cv2.imshow('sliding_window_detection', window_detection)
    
    if not found:
        #print("NOT FOUND: ", found)
        black_frame = np.zeros_like(frame).astype(np.uint8)
        return frame
    
    left_fit, right_fit = poly_fit(leftx, lefty, rightx, righty, binaryImage) #,output_show=True)           
    
    """LDW-STEP#5 SANITY CHECK"""
    sanity_stat=True
    #sanity_stat = sanity_check(leftx, rightx, left_fit, right_fit, 0, .25)

    """LDW-STEP#6 DEAL WITH RELIABLE & UNRELIABLE FIT"""
    if sanity_stat:
        # Save as last known reliable fit
        ref_left, ref_right = left_fit, right_fit
        left_fit_up, right_fit_up = left_fit, right_fit
    else:
        if len(leftx) == 0:
            right = right_fit[0] * 540 ** 2 + right_fit[1] * 540 + right_fit[2]
            left_fit_up = np.array([-0.0001, 0, right - 200])
            right_fit_up = right_fit

        elif len(rightx) == 0:
            left = left_fit[0] * 540 ** 2 + left_fit[1] * 540 + left_fit[2]
            right_fit_up = np.array([-0.0001, 0, left + 200])
            left_fit_up = left_fit

        else:
            left_fit_up, right_fit_up = ref_left, ref_right
    
    """LDW-STEP#7 TRIGGER WARNING"""
    midframe=300
    detection_value = warning(left_fit_up, right_fit_up,midframe=midframe)
    
    if detection_value == 1:
        ldw_output = detection_image[5]
        
    elif detection_value == 2:
        ldw_output = detection_image[6]
        
    elif detection_value == 0 or detection_value == 3:
        ldw_output = detection_image[4]

    cv2.imshow('LDW OUTPUT', ldw_output)
    
    '''''
    w_Key = cv2.waitKey(5)
    
    if w_Key%256 == 32:
          
        image_counter = image_counter + 1
        print('Saving File : ', image_counter)
        img_name1 = str(image_counter) + "_dewarped_color" + '.jpg'
        filename1 = os.path.join(save_path, img_name1) 
        #print('File Name : ', filename1)
        cv2.imwrite(filename1, dewarped_color)
        
        img_name2 = str(image_counter) + "_dewarped_binary" + '.jpg'
        filename2 = os.path.join(save_path, img_name2)
        #print('File Name : ', filename2)
        cv2.imwrite(filename2, warped_image_masked)
        
        img_name3 = str(image_counter) + "_window_detection" + '.jpg'
        filename3 = os.path.join(save_path, img_name3)
        #print('File Name : ', filename3)
        cv2.imwrite(filename3, window_detection)
        
        img_name4 = str(image_counter) + "_final_frame" + '.jpg'
        filename4 = os.path.join(save_path, img_name4)
        #print('File Name : ', filename4)
        cv2.imwrite(filename4, frame)
        
        json_file = os.path.join(save_path, 'diff_data.json')
        diff_dict = {image_counter:diff_value}
        
        with open(json_file, 'r+') as file:
            data = json.load(file)
            data.update(diff_dict)
            file.seek(0)
            json.dump(data,file)
        '''
    
    #return dewarped_image

if __name__ == '__main__':
    main()
