import os, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import json
from numpy import NaN, Inf, arange, isscalar, asarray, array
plt.ion()

fixed_scaled_frame_width = 0
fixed_scaled_frame_height = 0

l = r = croph = LV1 = LV2 = 0
LS1 = 0
US1 = 255
LS2 = 0
US2 = 255
UV1 = 255
UV2 = 255

crop_height = 0
number = 0

transformation_matrix = np.float32([[1, 0, 0], [0, 1, crop_height]])

rightRegionThresh = 366 
leftRegionThresh = 288
right_lane_flag = False
left_lane_flag = False
right_lane_counter = 0
left_lane_counter = 0
prev_detection = False
right_trigger = []
left_trigger = []
    
def read_calibration_value():
    global l, r, croph
    with open('transform_matrix.json', 'r') as json_file:
        data = json.load(json_file)
    
    l = data["l"]
    r = data["r"]
    croph = data["croph"]

def scale_to_fixed(frame):
    # Scale incoming image to 540x960
    global fixed_scaled_frame_height, fixed_scaled_frame_width

    scalewidth = frame.shape[1] / 1280
    scaleheight = frame.shape[0] / 720

    frame = cv2.resize(frame, (0, 0), fx=1 / 2 / scaleheight, fy=1 / 2 / scalewidth)
    (fixed_scaled_frame_height, fixed_scaled_frame_width) = frame.shape[:2]

    return frame


def getBrightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = cv2.mean(hsv)

    #cv2.mean() will return 3 numbers, one for each channel:
    #      0=hue
    #      1=saturation
    #      2=value (brightness)
    
    return result[2]


def compute_perspective_transform(frame, toEagleEye=True):
    global r, l, croph
    # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
    
    #print('r : ', r, ', l : ', l, ', croph : ', croph)
    
    eagleEyeRightSide = r
    eagleEyeLeftSide = l
    
    x1 = 0
    y1 = croph
    x2 = frame.shape[1] - 1
    y2 = croph
    x3 = 0
    y3 = frame.shape[0] * 0.9 - 1
    x4 = frame.shape[1] - 1
    y4 = frame.shape[0] * 0.9 - 1

    src = np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)], dtype="float32")
    W = frame.shape[1]
    L = frame.shape[0]

    dst = np.array([(0, 0), (W - 1, 0), (W / 2 + eagleEyeRightSide, L - 1), (W / 2 - eagleEyeLeftSide, L - 1)],
                   dtype="float32")
    if toEagleEye is True:
        M = cv2.getPerspectiveTransform(src, dst)

    elif toEagleEye is False:
        M = cv2.getPerspectiveTransform(dst, src)
        
    return M


def apply_perspective_transform(frame2, toWarp=True):
    global transformation_matrix

    if toWarp is True:
        transformation_matrix = compute_perspective_transform(frame2, toEagleEye=True)
        warped_image = cv2.warpPerspective(frame2, transformation_matrix, (frame2.shape[1], frame2.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
        #if calib is True:
        #    cv2.imshow("warped_image",warped_image)
        temp = warped_image
    elif toWarp is False:
        #cv2.imshow("warped_color_image",frame2)
        
        transformation_matrix = G = compute_perspective_transform(frame2, False)
        warped_image = cv2.warpPerspective(frame2, G, (frame2.shape[1], frame2.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
        #cv2.imshow("dewarped_color_image",frame2)
        #cv2.waitKey(0)
        temp = warped_image
        
    return temp  # warped_image


def sharpened(warped_image):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(warped_image, -1, kernel_sharpening)


    return sharpened

def compute_binary_image(color_image, LV1, LV2):
    global LS1, US1, LS2, US2, UV1, UV2
    # Convert to HLS color space and separate the S channel
    # Note: img is the undisted image
    hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)    

    boundaries = ([0, LS1, LV1], [179, US1, UV1])
    lower = np.array(boundaries[0], dtype=np.uint8)
    upper = np.array(boundaries[1], dtype=np.uint8)
    WnB1 = cv2.inRange(hsv, lower, upper)

    boundaries = ([80, LS2, LV2], [179, US2, UV2])
    lower = np.array(boundaries[0], dtype=np.uint8)
    upper = np.array(boundaries[1], dtype=np.uint8)
    WnB2 = cv2.inRange(hsv, lower, upper)

    combined_w_Y = WnB1 | WnB2

    return combined_w_Y


def edge_filter(cropped_image, binary_frame):
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 50
    thresh_max = 100
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 255) & (sxbinary == 255)] = 255
    combined_binary = cv2.bitwise_or(binary_frame, sxbinary, combined_binary)

    return combined_binary

def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    # if len(v) != len(x):
    #     sys.exit('Input vectors v and x must have same length')
    #
    # if not isscalar(delta):
    #     sys.exit('Input argument delta must be a scalar')
    #
    # if delta <= 0:
    #     sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def get_max_by_col(li, col):
    # col - 1 is used to 'hide' the fact lists' indexes are zero-based from the caller
    return max(li, key=lambda x: x[col - 1])[col - 1]

def extract_lanes_pixels(binary_warped, plot_show=False):
    #bool to check if the left and right lane exists
    leftLaneFound = False
    rightLaneFound = False
    
    # Set the width of the windows +/- margin
    margin = 20

    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Choose the number of sliding windows
    nwindows = 20
    # Take a histogram of the bottom 2/3 of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2/3):, :], axis=0)  # img.shape[0] to get image height
    #print('histogram : ' + histogram)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    maxTab, minTab = peakdet(histogram, 3000)
    midpoint = np.int(histogram.shape[0] / 2)
    #print("maxTab",maxTab)
    
    if len(maxTab) == 0:
        return None, None, None, None, False, False, False
    
    maxTab = maxTab[maxTab[:,1]>0] #10000
    #maxTab = maxTab[maxTab[:,1]>500]
    maxTabLocations = maxTab[:,0] #slice the fit column only
    leftHandLocations = maxTabLocations[maxTabLocations < midpoint]
    rightHandLocations = maxTabLocations[maxTabLocations > midpoint]
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if len(leftHandLocations) == 0: #check if it found any lane
        leftLaneFound = False
    else:
        leftLaneFound = True
        leftx_base = leftHandLocations[-1] #rightmost of the left locations
    if len(rightHandLocations) == 0:
        rightLaneFound = False
    else:
        rightLaneFound = True
        rightx_base = rightHandLocations[0] #leftmost of the right locations  
    if rightLaneFound == False or leftLaneFound == False: #couldnt find any lanes
        return None, None, None, None, False, False, False
    if plot_show:
        # for loc in maxTab:
        #     plt.plot([loc[0], loc[0]], [0, 10000], 'k-')  # x,x then y,y
        #plt.plot([rightx_base, rightx_base], [0, 10000], 'k-') #x,x then y,y
        #plt.plot([leftx_base, leftx_base], [0, 10000], 'k-')
        plt.plot(histogram)
        plt.pause(0.0001)
        plt.show()
        plt.gcf().clear()

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    #leftx_current = leftx_base
    #rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    #left_lane_inds = []
    #right_lane_inds = []
    # Create 3 channels to draw green rectangle
    out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("out_image", out_img)
    
    leftx,lefty, is_window_detected, out_img = slideWindows(binary_warped,leftx_base,nonzerox,nonzeroy,out_img=out_img)
    rightx,righty, is_window_detected, out_img = slideWindows(binary_warped, rightx_base,nonzerox,nonzeroy,out_img=out_img)

    # Concatenate the arrays of indices
    #left_lane_inds = np.concatenate(left_lane_inds)
    #right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    #leftx = nonzerox[left_lane_inds]
    #lefty = nonzeroy[left_lane_inds]
    #rightx = nonzerox[right_lane_inds]
    #righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, True, True, is_window_detected #, left_lane_inds, right_lane_inds

def slideWindows(image,basex_current,nonzerox,nonzeroy,nwindows=20,margin = 20,minpix = 100, out_img=None):
    window_height = np.int(image.shape[0] / nwindows)
    #lane_inds = []
    pointX = []
    pointY = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_x_low = int(basex_current - margin)
        win_x_high = int(basex_current + margin)
        #win_xright_low = int(rightx_current - margin)
        #win_xright_high = int(rightx_current + margin)

        # distance = win_xright_high - win_xleft_high
        # print("dist", distance) #140 is good distance

        # Draw the windows on the visualization image   
        #if out_img is not None:
            #cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        # https://stackoverflow.com/questions/7924033/understanding-numpys-nonzero-function
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        #print ("good inds: ", good_inds)
        # Append these indices to the lists
        #lane_inds.append(good_inds)
        #right_lane_inds.append(good_right_inds)
        
        pointX.append(int(win_x_high+win_x_low)/2)
        pointY.append(int(win_y_high+win_y_low)/2)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            basex_current = np.int(np.mean(nonzerox[good_inds]))
        #if len(good_right_inds) > minpix:
        #    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #if calib is True:
    #cv2.imshow("window detect", out_img)   // enable to "# Draw the windows on the visualization image"
    #cv2.waitKey(0)
    
    if out_img is not None:
        return pointX, pointY, True, out_img

    else:
        return pointX, pointY, False, out_img

def poly_fit(leftx, lefty, rightx, righty,binary_warped): #, output_show=False):
    # Fit a second order polynomial to each
    global left_fit, right_fit
    h = 360
    try:
        if len(leftx) != 0 and len(rightx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        elif len(leftx) == 0:
            right_fit = np.polyfit(righty, rightx, 2)
            right = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
            left_fit = np.array([-0.0001, 0, right - 200])
        elif len(rightx) == 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            left = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
            right_fit = np.array([-0.0001, 0, left + 200])

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        #nonzeroy = np.array(nonzero[0])
        #nonzerox = np.array(nonzero[1])
        #print("nonzerox: ", nonzerox)

    except Exception as e:
        print(e)
        # pass

    return left_fit, right_fit

def warning(left_fit, right_fit):
    global number, leftRegionThresh, rightRegionThresh, right_lane_flag, left_lane_flag, prev_detection, right_lane_counter, left_lane_counter, left_trigger, right_trigger

    h2 = 360
    yl = left_fit[0] * h2 ** 2 + left_fit[1] * abs(h2) + left_fit[2]
    #print("yl", yl)
    yr = right_fit[0] * h2 ** 2 + right_fit[1] * abs(h2) + right_fit[2]
    #print("yr : ", yr, " & yl : ", yl)

    diffLeftRight = yr - yl
    
    if 100 < diffLeftRight < 270: 
        # 2
        # check raw detection
        if leftRegionThresh < yr < rightRegionThresh:
            print('RIGHT')
            right_trigger.append('1') 
            
            if len(right_trigger) > 2:
                if left_lane_flag is not True:  # check right real detection 
                    right_lane_counter = 5
                elif left_lane_flag is True:
                    left_lane_counter = 5
            
        elif leftRegionThresh < yl < rightRegionThresh:
            print('LEFT')
            left_trigger.append('1')
            
            if len(left_trigger) > 2:
                if right_lane_flag is not True: # check left real detection
                    left_lane_counter = 5
                elif right_lane_flag is True:
                    right_lane_counter = 5
                    
        else:
            if right_trigger:
                right_trigger.pop()
            elif left_trigger:
                left_trigger.pop() 
                
            

        # counter
        if right_lane_counter > 0:  #triggered after real detection
            right_lane_counter -= 1
            right_lane_flag = True
           
        else: 
            if right_lane_flag is True:
                right_trigger.clear()
                left_trigger.clear()
            right_lane_flag = False
           

        if left_lane_counter > 0:   #triggered after real detection
            left_lane_counter -= 1
            left_lane_flag = True
           
        else:
            if left_lane_flag is True:
                right_trigger.clear()
                left_trigger.clear()
            left_lane_flag = False
           
        print('Right : ', right_trigger, ' ,,,  Left : ', left_trigger)
        
        # final output warning
        if right_lane_flag is True:
            number = 1
                
        elif left_lane_flag is True:
            number = 2
        
        else:
            number = 0 

        # right alert : number = 1
        # left alert : number = 2
        # normal or no detection : number = 0
        return number

    else:
        #print("error")
        #number = 3
        
        number = checkPrevDetection()
        
        return number


def checkPrevDetection():
    
    global right_lane_flag, left_lane_flag, left_lane_counter, right_lane_counter, prev_detection, left_trigger, right_trigger           
    
    if right_trigger:
        right_trigger.pop()
    elif left_trigger:
        left_trigger.pop() 
    
    # counter
    if right_lane_counter > 0:  #triggered after real detection
        right_lane_counter -= 1
        right_lane_flag = True
           
    else: 
        if right_lane_flag is True:
            right_trigger.clear()
        right_lane_flag = False
           

    if left_lane_counter > 0:   #triggered after real detection
        left_lane_counter -= 1
        left_lane_flag = True
           
    else:
        if left_lane_flag is True:
            left_trigger.clear()
        left_lane_flag = False
           
    print('Right : ', right_trigger, ' ,,,  Left : ', left_trigger)
     
    # final output warning
    if right_lane_flag is True:
        if prev_detection is True:
            number = 1
        else: 
            number = 0
            prev_detection = True
                
    elif left_lane_flag is True:
        if prev_detection is True:
            number = 2
        else: 
            number = 0
            prev_detection = True
    else:
        number = 0 
        prev_detection = False

    return number

