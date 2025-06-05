import cv2
import math
import numpy as np
from tracker import *

backsub = cv2.createBackgroundSubtractorMOG2()  
#backsub = cv2.bgsegm.createBackgroundSubtractorMOG()  
cap = cv2.VideoCapture('videos/vidrio60.mp4') 
#cap = cv2.VideoCapture(0) 
i = 0
minArea = 5
font = cv2.FONT_HERSHEY_SIMPLEX
band= False
 
tracker = EuclideanDistTracker()

def checkStatus(self, x, y):
    if band == True:
        x=1
    #nom = abs(self.a * x + self.b * y + self.c)
    return band

# ret, frame = cap.read()
# #area_pts = cv2.selectROI("Frame", frame, fromCenter=False,  showCrosshair=True)

# TopLeft = (area_pts[0],area_pts[1])
# TopRight = ((area_pts[0]+ area_pts[2]),area_pts[1])
# BotRight = ((area_pts[0]+ area_pts[2]), (area_pts[1]+ area_pts[3]))
# BotLeft = (area_pts[0],(area_pts[1]+ area_pts[3]))
#aux = np.zeros((len(frame), len(frame[0]), 3))
#aux.fill(255) # or img[:] = 255
while True:
    ret, frame = cap.read()
    #cap.set(cv2.CAP_PROP_FPS, 1)
   # cv2.putText(frame, "fps: " + str(()), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # frame[160,160] = [0,255,255]
    # frame[161,160] = [0,255,255]
    negro = np.array([[150, 150], [195,150], [195,165], [150,165]]) 
    negro2 = np.array([[300, 115], [325,115], [325,150], [300,150]]) 
   # negro = np.array([[100, 100], [300,100], [300,200], [100,200]]) 
    cv2.fillPoly(frame, [negro], color=[0,0,0])
    cv2.fillPoly(frame, [negro2], color=[0,0,0])
   # negro[] = [0,255,0]
   # fgmask = backsub.apply(frame, None, 0.01)
    area_pts = np.array([[10, 100], [350,100], [350,350], [10,350]])
   
    #area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)
    fgmask = backsub.apply(image_area)
    
    
    # Remove border
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    temp1 = 255 - cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    temp2 = 255 - cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)
    result = cv2.add(temp3, fgmask)

    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # thresh=cv2.dilate(thresh,None,iterations=1)
    
    
    
    
    
    
    #erode = cv2.erode(fgmask, None, iterations=2)      
    #dilate = cv2.dilate(fgmask, None, iterations=1)  
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
   # result = np.zeros((fgmask.shape), np.uint8)
   # result = np.zeros((360,640,3), np.uint8)

    for i in range(0, nlabels - 1):
        #if areas[i] >= 100:   #keep
        if areas[i] >= 5:   #keep
            result[labels == i + 1] = 255

    deteccions = []
    
    lines = cv2.HoughLinesP(fgmask, 1 , np.pi/180, 100, minLineLength=50,maxLineGap=80)
    if lines is not None:
        cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        #for line in lines:
        x1,y1,x2,y2 = lines[0][0]
        #cv2.waitKey()
        
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.line(result,(x1,y1),(x2,y2),(255,255,255),2)
        
   
        
        #ancho = x2 - x1
        #x,y,w,h = cv2.boundingRect(line[0])
        # cv2.line(result,(x1,y1),(x2,y2),(255,255,255),1)
        #color = np.random.randint[255,0,0]
        # r = np.random.randint(0, 255)
        # g = np.random.randint(0, 255)
        # b = np.random.randint(0, 255)
        # rand_color = (r, g, b)
            #color=[255,0,0]
            #random.shuffle(color)
	# colors[0] = [0,0,0]
	# colored_components = colors[labels]
	# cv2.imshow('output', gray)
	# cv2.imshow('colored', colored_components)
        #cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        #cv2.line(frame,(x1,y1),(x2,y2),(rand_color),2)
            #M = cv2.moments(line[0])
            #cX = int(M["m10"] / M["m00"])
            # #The center point is simply (cX, cY) and you can draw this with cv2.circle()

       # cv2.circle(frame, (int(ancho/2), y1), 3, (0, 255, 0), -1)
        #box = cv2.boxPoints(rect)
           # box = np.int_(box)
           # center = (x+w//2, y+h//2)
        # ancho= x2-x1
        # center = x2//2
        #cv2.circle(frame, (int(center), y2), 3, (0, 255, 255), 2)
        
        # deteccions.append([x1, y1, ancho, 2])
            #deteccions.append(line[0])
            #deteccions.append([cX, cY, ancho, 2])
            #deteccions.append([M])
            #deteccions.append([extLeft[0], extLeft[1], w, h])
    # boxers_ids= tracker.update(deteccions)
            
    # for box_id in boxers_ids:
    #     x, y, w, h, id = box_id
    #     cv2.putText(frame,  str(id) , (int(x), int(y)),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    #     # cv2.drawContours(frame,[box], -1,(0,255,0), 3)
    #     # cv2.drawContours(frame,[rect], -1,(0,255,0), 3)
    #        # cv2.waitKey()
    
    #cv2.waitKey()
    #contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #  contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 3)
    
    
    
    # maxContour = 0
    # for contour in contours:
    #     contourSize = cv2.contourArea(contour)
    #     if contourSize > maxContour:
    #         maxContour = contourSize
    #         maxContourData = contour #if len(contours)>2:
    #        # cv2.drawContours(frame, [maxContourData], -1, (0, 255, 255), 3)
           # print(maxContour)
            
    # cv2.putText(frame, "contour: " + str(maxContour), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
    #area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])

    #cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    #cv2.rectangle(frame,(10,10),(frame.shape[1],40),(0,0,0),-1)
    #image_area = cv2.bitwise_and(gray, gray, mask=imAux)
    #image_area = cv2.bitwise_and(fgmask, fgmask, mask=imAux)

    #fgmask = fgbg.apply(image_area)
    #fgmask = fgbg.apply(image_area)
    #mask = object_detector.apply(frame)
    
    #fgmask = object_detector.apply(image_area)
    cv2.putText(frame, "contour: " + str(len(contours)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    #contours = contours[0] if len(contours) == 2 else contours[1]
    for cnt in contours:
        #cv2.waitKey()
       # print(x, y, w, h) = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 340:
            print(cv2.contourArea(cnt))
            #aux = aux.copy()
           # cv2.drawContours(aux, contours, -1, (0, 0, 0), 1)
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
			#perimeter = cv2.arcLength(cnt, True)
			#perimeter = cv2.arcLength(cnt, False)
			# approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
			# cv2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
			# (x, y, w, h) = convexHull
			# (x, y, w, h) = cv2.boundingRect(convexHull)
			#(x, y, w, h) = cv2.boundingRect(cnt)
           #moments = cv2.moments(erode, True)               
            x, y, w, h = cv2.boundingRect(cnt)
            moments = cv2.moments(cnt)               
          #  moments = cv2.moments(cnt, True)               
            area = moments['m00']
            
            #centerX = 0
            #centerY = 0
            #(x, y, w, h) = cv2.boundingRect(cnt)
            #rect = cv2.minAreaRect(cnt)
            #box = cv2.boxPoints(rect)
           #box = np.int_(box)
           # cv2.waitKey()
            #cv2.rectangle(frame, (x,y), [box], (0,255,0), 3)
            #cv2.drawContours(frame,box,0,(0,255,0),1)
    #cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
            
            rect = cv2.minAreaRect(cnt)
            centerX = rect[0][0]
            centerY = rect[0][1]
            cv2.circle(frame, (int(centerX), int(centerY)), 3, (0, 0,255), 3)
            #cv2.putText(frame,   str(centerX) , (int(centerX), int(centerY)),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
            extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
          #  cv2.putText(frame,   str(extRight[0]) , (int(extRight[0]), int(extTop[0])),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
          #  cv2.circle(frame, (int(extLeft[0]), int(extLeft[1])), 2, (255, 0,0), 3)
          #  cv2.circle(frame, (int(extRight[0]), int(extRight[1])), 2, (255, 0,0), 3)
           # cv2.waitKey()
            
            #box = cv2.boxPoints(rect)
           # box = np.int_(box)
            deteccions.append([centerX, centerY, w, h])
            #deteccions.append([extLeft[0], extLeft[1], w, h])
            boxers_ids= tracker.update(deteccions)
            
            for box_id in boxers_ids:
                x, y, w, h, id = box_id
                cv2.putText(frame,  str(id) , (int(x), int(y)),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
           # cv2.drawContours(frame,[box], -1,(0,255,0), 3)
           # cv2.drawContours(frame,[rect], -1,(0,255,0), 3)
           # cv2.waitKey()
            
        #    # if centerX >= 270 and centerX <= 290 and centerY >= 100 and centerY <= 300:
        #     if int(extRight[0]) >= 275 and int(extRight[0]) <= 285 and centerY >= 100 and centerY <= 300:
        #         band = True      
        #         i += 1
               # cv2.line(frame, (280, 130),(280, 300), (0, 255, 0), 1)
               # print(i)
               # print("area" + str(area))
            #else:
            
            #break
            # else:
            #     cv2.putText(frame,  str(centerX) , (int(x), int(y)),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            
                #cv2.line(frame, (280, 130),(280, 300), (0, 0, 255),2)
           # cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
            # if moments["m00"] != 0:
            #     cX = int(moments["m10"] /  moments["m00"]) 
            #     cY = int(moments["m01"] / moments["m00"])
            #    # cv2.circle(frame, (int(cX), int(cY)), 3, (0, 255, 255), 2)

            #     #cv2.waitKey()
            #if moments['m00'] >= minArea:
           # cv2.waitKey()

            #x = (moments['m10'] // moments['m00'])
           # y = (moments['m01'] // moments['m00'])
        #         #counter =id
        #         #cv2.putText(frame,  str(id) , (x, y),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
		# #counter_actual=id
            #print("x:" + str(x)  + "y:"  + str(y))
            # if band == True:
            #     if x >= 270 and x <= 290 and y >= 100 and y <= 300:
            #         i += 1
            #         cv2.line(frame, (280, 130),(280, 300), (0, 255, 0),2)
            #         print(i)
            #         print("area" + str(area))
            #     else:
            #         cv2.line(frame, (280, 130),(280, 300), (0, 0, 255),2)
                    
           # else:
            
    # if band == True:
    #     #cv2.line(frame, (280, 130),(280, 300), (0, 255, 0), 2)
    # else:
        #cv2.line(frame, (280, 130),(280, 300), (0, 0, 255),2)
    
    cv2.drawContours(frame, [area_pts], -1, (255,0,255), 2)
    cv2.putText(frame,'COUNT: %r' %i, (10,30), font, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    #cv2.imshow("aux", aux)
    cv2.imshow("result", result)
        
            
   # cv2.line(frame, (300,0), (300,500), (0, 255, 0), 3)

    cv2.imshow("background sub", fgmask)
    
    #cv2.waitKey()
   # cv2.imshow("Track", frame)
    key = cv2.waitKey(100)
    if key is ord('q'):
        break