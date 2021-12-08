from final_ldw_helper_functions import *
import numpy as np
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

DAY_BRIGHTNESS_TRESH = 100

def run_pipeline(frame, mask_img):
    global ref_left, ref_right, DAY_BRIGHTNESS_TRESH

    brightness = getBrightness(frame)
    warped_image = apply_perspective_transform(frame, toWarp=True)
    off_lane = False 
    
    if brightness > DAY_BRIGHTNESS_TRESH: 
        time_str = 'Time : DAY'
        # ---------LDW PIPELINE
        """LDW-STEP#1 CROP ROAD ROI"""
        '''
        dewarped_color = apply_perspective_transform(frame.copy(), toWarp=True)
        cv2.line(dewarped_color,(288,0),(288,620),(0,255,0),2)
        cv2.line(dewarped_color,(366,0),(366,620),(0,255,0),2)
        cv2.imshow("dewarped_color", dewarped_color)
        '''
        
        binary_image = compute_binary_image(warped_image, 230, 156)
        whites = cv2.countNonZero(binary_image)
        #print('Whites : ', whites)
        
        if whites < 1000:
            off_lane = True
            
        """LDW-STEP#2 PERSPECTIVE TRANSFORM"""
        binaryImage = edge_filter(warped_image, binary_image)
        binary_image_masked = cv2.bitwise_and(binaryImage, mask_img)
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        binary_image_masked = cv2.morphologyEx(binary_image_masked, cv2.MORPH_CLOSE, kernel_2)
        #cv2.imshow("binary_image_masked", binary_image_masked)
        #cv2.putText(binary_image_masked, str(time_str), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255), 2, cv2.LINE_AA)

    ############ night #############
    else:
        
        if brightness > 36 and brightness < 50:
            alpha = 1.7
            beta = -30
            frame = cv2.addWeighted(frame,alpha,np.zeros(frame.shape,frame.dtype),0,beta)       
        
        #time_str = 'Time : NIGHT'
        '''
        dewarped_color = warped_image.copy()
        cv2.line(dewarped_color,(288,0),(288,620),(0,255,0),2)
        cv2.line(dewarped_color,(366,0),(366,620),(0,255,0),2)
        cv2.imshow("dewarped_color", dewarped_color)
        '''
        sharpened_image = sharpened(warped_image)
        binaryImage = compute_binary_image(sharpened_image, 196, 214)
    
        binaryImage = edge_filter(warped_image, binaryImage)
        binary_image_masked = cv2.bitwise_and(binaryImage, mask_img)
        #cv2.imshow("binary_image_masked", binary_image_masked)
        #whites = cv2.countNonZero(binary_image_masked)
        #print('Whites : ', whites)

        #cv2.putText(binary_image_masked, str(time_str), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255), 2, cv2.LINE_AA)
        
    """LDW-STEP#4 DEFINE LANE-LINES"""
    leftx, lefty, rightx, righty, found, maxTab_value, is_window_detected = extract_lanes_pixels(binary_image_masked, plot_show=False)
    
    if not found or not maxTab_value or not is_window_detected or off_lane is True:
        detection_value = checkPrevDetection()
        return detection_value
    
    else:
        left_fit, right_fit = poly_fit(leftx, lefty, rightx, righty, binary_image_masked)          
    
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
    detection_value = warning(left_fit_up, right_fit_up)
    
    return detection_value
