import cv2
import math
import numpy as np
from tracker import *
# Import the time library
import time
#import pyodbc
import imutils
import os
from datetime import datetime
import tkinter as tk


backsub = cv2.bgsegm.createBackgroundSubtractorMOG()  
#backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  
#cap = cv2.VideoCapture(0) 
#cap = cv2.VideoCapture(1) 
cap = cv2.VideoCapture('videos/vidrio71.mp4') 

f = 0
fpLast = 0
bitvalue = 0
counter = 0
counter2 = 0
j = 0
k = 0
m = 0
minArea = 20
value = 0
pw = 0

font = cv2.FONT_HERSHEY_SIMPLEX;

band=False
bandAzul = False
bandVerde = False
bandMorado = False
bandStart = False
bandRecord = False
bandON= False
bandCaptura= False
#global counter = 0;
tracker = EuclideanDistTracker()

now = datetime.now()
timeON = datetime.now()
#filename = now.strftime("%H:%M:%S")
filename = now.strftime("%Y-%m-%d_%H_%M_%S")

#Datos = 'objects' + str(filename)
#Datos = 'objects' + str(filename)
Datos = 'objects' 
#Datos = '\\\\10.18.172.30\\Departments\\IT\\Read\\Reportes_Tickets\\objects' + str(filename)
if not os.path.exists(Datos):
    print('Carpeta creada: ',Datos)
    os.makedirs(Datos)
    
counter=0
#print('Enter min value:')
#minvalue = input()


#from Guardar import *
def timer1():
    # Calculate the start time
    start = time.time()

    running = True
    seconds = 0
    end = 1

    while (running):
        time.sleep(1)
        seconds +=1
        if seconds >= end:
            running = False
            
                
def BandValue(bitvalue):
    
	global bandVerde
	global bandAzul
	global bandStart
	global bandMorado
	global bandRecord
	global bandCaptura
	global counter
	global bandON
	global j
	global k
	global m
	global pw
	global timestampON
	global timestampOFF
    #version 3
    
    
	if  bitvalue == 1:
		
		j += 1
		k = 0
  
		bandON = True
  
		if j == 1:
			timestampON = datetime.now()
  
		if j == 10:
			
			bandVerde = True
			m += 1
			
			pw = j
			#bandRecord = True
   

			#if m >= 1:
			#bandStart = True
   
		#if j == 9:
            
	else:
        #bitvalue es cero
   
		k +=1
        
		# if k==1:
				
		
		if k == 1:
			if bandON == True:
				timestampOFF = datetime.now()
				bandCaptura = True
				pw = j
				j = 0 
   
		if k >= 4:
			#pw = j

			if bandVerde == True:
           
				counter +=1

				bandAzul = True

				bandVerde = False

				m=0
				k=0

            
            #cv2.putText(frame,'COUNT: %r' %counter, (400,90), font, 1, (0, 255, 0), 2)
          
            #cv2.putText(frame, "fp: " + str(f), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(frame,  "j: " +str(j), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
           
            
    #version 1
    # if  bitvalue == 1:
        
    #     bandVerde = True
                            
    # else:
        
    #     if bandVerde == True:
            
    #         counter +=1
            
    #         bandVerde = False
            
    #     else:
            
    #         bandVerde = False   
    
    #version 2
        
    # if  bitvalue == 1:
    #     if  bandAzul == True:
    #         bandVerde = True
    #     else:
    #         bandAzul = True                   
        
    # else:
    #     if bandAzul == True:
    #         bandAzul =False
            
    #     else:
            
    #         if bandVerde == True:
            
    #             counter +=1
            
    #             bandVerde = False
    #             bandAzul = False
                
               
                #Unix Timestamp: {time.time()}'
					# #print("Imagen impresa: "+ str(i))
					# if 290 > distancia > 260:
         
					# 	cv2.imwrite(Datos+'/objeto_{}.jpg'.format(formatted),objeto) 
						# counter +=1
						# j+=1
            #else:
            
                #bandVerde = False
             
   # print("contador:"  + str(counter))
    #return bandAzul
            
# def checkStatus(self, x, y):
#     if band == True:
#         x=1
#     #nom = abs(self.a * x + self.b * y + self.c)
#     return band
def nothing(pos):
		pass

ret, frame = cap.read()
# area_pts = cv2.selectROI("area_pts", frame, fromCenter=False,  showCrosshair=True)

# TopLeft = (area_pts[0],area_pts[1])
# TopRight = ((area_pts[0]+ area_pts[2]),area_pts[1])
# BotRight = ((area_pts[0]+ area_pts[2]), (area_pts[1]+ area_pts[3]))
# BotLeft = (area_pts[0],(area_pts[1]+ area_pts[3]))
TopLeft = [272, 175]
TopRight = [304, 175]
BotRight = [304, 383]
BotLeft = [272, 383]
area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])

#print(area_pts)
# button dimensions (y1,y2,x1,x2)
#cv2.rectangle(frame, (100,100) , cnt ,(0,255,0),-1)
#cv2.rectangle(frame, (20,60) , (50,250) ,(0,255,0),-1)
#cv2.rectangle[:button[1],button[2]:button[3]] = 180
#cv2.putText(frame, 'Button',(100,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)

button = [20,60,50,250]
# function that handles the mousclicks
def process_click(event, x, y,flags, params):
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            print("reset counter")
            global counter
            counter=0
            
            now = datetime.now()
            filename = now.strftime("%Y-%m-%d_%H_%M_%S")
            #Datos = 'objects' + str(filename)
            Datos = 'objects' 
            if not os.path.exists(Datos):
                print('Carpeta creada: ',Datos)
                os.makedirs(Datos)
    
cv2.namedWindow('Control')
cv2.setMouseCallback('Control',process_click)
# create button image
control_image = np.zeros((80,300), np.uint8)
control_image[button[0]:button[1],button[2]:button[3]] = 180
cv2.putText(control_image, 'Reset',(100,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
# create a window and attach a mousecallback and a trackbar
cv2.imshow('Control', control_image)
        
#cv2.createTrackbar("Capture", 'Control', 0,1, startCapture)
# Delete_area = cv2.selectROI("Delete_area", frame, fromCenter=False,  showCrosshair=True)
# #Delete_area = np.array([[150, 150], [195,150], [195,165], [150,165]]) 
# TopLeft_D = (Delete_area[0],Delete_area[1])
# TopRight_D= ((Delete_area[0]+ Delete_area[2]),Delete_area[1])
# BotRight_D = ((Delete_area[0]+ Delete_area[2]), (Delete_area[1]+ Delete_area[3]))
# BotLeft_D = (Delete_area[0],(Delete_area[1]+ Delete_area[3]))
#Delete_area = np.array([TopLeft, TopRight, BotRight , BotLeft])

#area_pts = np.array([[50, 110], [800,110], [800,600], [50,600]])
#Delete_area = np.array([[TopLeft_D], [TopRight_D], [BotRight_D] , [BotLeft_D]])
# negro = np.array([[290, 280], [320,280], [320,320], [290,320]]) 
# negro1 = np.array([[230, 290], [234,290], [234,293], [230,293]]) 
# negro2 = np.array([[255, 150], [260,150], [260,160], [255,160]]) 
# negro3 = np.array([[260, 180], [265,180], [265,184], [260,184]]) 
# negro5 = np.array([[260, 150], [265,150], [265,160], [260,160]]) 
# negro4 = np.array([[310, 140], [320,140], [320,155], [310,155]]) 
# negro6 = np.array([[248, 190], [252, 195], [252, 195], [248, 190]]) 
# negro7= np.array([[228, 175], [235, 180], [235, 180], [228, 175]]) 
# negro8= np.array([[120, 160], [200, 160], [200, 220], [120, 220]]) 

kernel = np.ones((1,1),np.uint8)

#cv2.namedWindow('Thresholds')
# cv2.createTrackbar('min','thresh',0,255, nothing)
# cv2.createTrackbar('max','thresh',255,255, nothing)
# # cv2.createTrackbar('area','Thresholds',0,400, nothing)
# cv2.createTrackbar('lowc','Thresholds',100,400, nothing)
# cv2.createTrackbar('maxc','Thresholds',200,400, nothing)

def Leer():
	global value
	global f
	global bandAzul
	global j
	global Start
	global bandStart
	global bandMorado
	global ON_formatted
	global OFF_formatted 
	global delay
	global fpLast
	global band
	global counter2
	global pw
	global bandON
	global bandCaptura
	global timestampON
	global timestampOFF
	timeON = datetime.now()

	
    
	while(True):
		# 
		f +=1

		ret, frame = cap.read()
		# cv2.fillPoly(frame, [negro], color=[0,0,0])
		#frGray_area_pts = np.array([[50, 110], [800,110], [800,600], [50,600]])
		# Cropping an image
		frGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#cropped_image = frGray[80:280, 150:330]
		#imAuxGray = np.zeros(shape=(frGray.shape[:2]), dtype=np.uint8)
		
		#area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])
		#area_pts = np.array([[250, 110], [450,110], [450,330], [250,330]])
		#area_pts = np.array([[280, 110], [300,110], [300,330], [280,330]])
        
		imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
		imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
		image_area = cv2.bitwise_and(frame, frame, mask=imAux)

		# min = cv2.getTrackbarPos('min','thresh')
		# max = cv2.getTrackbarPos('max','thresh')
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fgmask = cv2.GaussianBlur(gray, (5, 5), 0)
		#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel)
		#fgbinario = 
		fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
		#thresh = cv2.threshold(fgmask, min, max, cv2.THRESH_BINARY)[1]
		fgmask = backsub.apply(image_area)
   
		#erode = cv2.erode(fgmask, None, iterations=2)      
		#dilate = cv2.dilate(fgmask, None, iterations=1)  
		# do connected components processing
		nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

		#get CC_STAT_AREA component as stats[label, COLUMN] 
		areas = stats[1:,cv2.CC_STAT_AREA]

		result = np.zeros((labels.shape), np.uint8)
		# result = np.zeros((fgmask.shape), np.uint8)
		# result = np.zeros((360,640,3), np.uint8)

		for i in range(0, nlabels - 1):
			#if areas[i] >= 100:   #keep
			if areas[i] >= 5:   #keep
				result[labels == i + 1] = 255

		# deteccions = []
		# lines = cv2.HoughLinesP(thresh, 1 , np.pi/180, 100, minLineLength=5,maxLineGap=80)
		# if lines is not None:
		#     cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		# #for line in lines:
		#     x1,y1,x2,y2 = lines[0][0]

		#     #cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
		#     cv2.line(result,(x1,y1),(x2,y2),(255,255,255),1)
			
		contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cv2.drawContours(frame, contours , -1,(0,255,255), 3)
        
        #contours = contours[0] if len(contours) == 2 else contours[1]
		# global minvalue
		# whitePixelCount = cv2.countNonZero(result)
		# # blackPixelCount = cv2.countze(result)
		# #cv2.waitKey()
		# #if whitePixelCount< int(minvalue):
		# # if f >= 220:
		# #     #cv2.waitKey()
		# #     print("nada")
		# if whitePixelCount <= minArea:
		# 	value = 0
		# else:
		# 	value = 1
		# #print(str(whitePixelCount) + " " + str(value))
		#cv2.waitKey()
		#BandValue(value)
		#print("frame: " + str(f))

		#value = 0
		#cv2.putText(frame, str(len(contours)), (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		value = 0
		#if len(contours) > 2:
   
		for cnt in contours:
			for point in cnt:
				row, col = point[0]
			#cv2.waitKey()
	
			#print("_ _ _ _ _ _")
			#print("Area: " + str(cv2.contourArea(cnt)))
			#extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
				#cv2.circle(frame, int(row), 4, (255, 255, 0), thickness = -1)	

				#cv2.line(frame, (row, 180), (row, 380), (0, 255, 255), 1)
				#cv2.line(frame, (400, 80), (400, 300), (0, 0, 255), 2)
					#cv2.putText(frame, str(extLeft[0]), (extLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

				#if  extLeft[0] >= 250 and 300 >= extLeft[0]:
				if  row >= 250 and 300 >= row:
					value = 1
					#print("TRUE:" + str(row))
		
					#cv2.line(frame, (255, 80), (255, 300), (0, 255, 0), 2)
		
				#else:
					#if  row >= 420 and 450 >= row:
					#value = 0
					#print("FALSE:" + str(row))
     
				#if band == True:
	
		BandValue(value)
  
			#cv2.circle(frame, point[0], 4, (0, 255, 0), thickness=-1)	
				#print(row)
				#cv2.line(frame, (row, 100), (row, 300), (0, 255, 255),2)
				#extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
     
				#if  extLeft[0]>= 250 and 260 >= extLeft[0]:
					#bandMorado = True
					#print(extLeft[0])
					#print(extLeft[0])
					#cv2.putText(frame, str(extLeft[0]), (extLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
					#band = True
					#cv2.putText(frame, str(band), (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
     
					#cv2.putText(frame,  str(band), (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
						#if  row>= 370 and 375 >= row:
					# if  row>= 240 and 300 >= row:
					# 	print("FALSE:" + str(row))
		
						#print(row)
						#cv2.line(frame, (370, 80), (370, 300), (0, 255, 0), 2)

						#band = False
						#cv2.waitKey()
						#bandMorado = False
						#counter2 +=1
   
		# if len(contours)==0:
		# 	BandValue2(value)
	
		
				#cv2.putText(frame, str(extLeft[0]), (extLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
							#cv2.line(frame, (280, 100), (280, 300), (0, 255, 0),3)
       
		#cv2.putText(frame, str(bandMorado), (200,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
				#cv2.putText(frame, str(band), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        
					

					#cv2.imshow("frame", frame)
					#cv2.waitKey()
				#cv2.waitKey()
    
				#print(row, col)
				#pixel_counts[col] += 1

   
        #bandRoja = BandValue(value)
        #cv2.waitKey()
            
        #for cnt in contours:
            #cv2.waitKey()
           # print(cv2.contourArea(cnt))
            #if cv2.contourArea(cnt) > 70:
               # convexHull = cv2.convexHull(cnt)
               # cv2.drawContours(frame, [convexHull], -1, (0, 255, 0), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
                
            #x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(img_draw, (x, y), (x+w, y+h), color_bbox, 5)
            #cv2.rectangle(frame, (100,100) , cnt ,(0,255,0),-1)
           # cv2.rectangle(frame,int(x),int(y), [cnt], (0,255,0), 2)
            #cv2.putText(frame, str(int(x)) + ","+ str(int(y)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            #     if len(contours)==0:
            #     print(cv2.contourArea(cnt))
            # BandValue(value)
            # else:
            #     value = 0
                #print(cv2.contourArea(cnt))
                #aux = aux.copy()
            # cv2.drawContours(aux, contours, -1, (0, 0, 0), 1)
                #cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
                #perimeter = cv2.arcLength(cnt, True)
                #perimeter = cv2.arcLength(cnt, False)
                # approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                # cv2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
                # (x, y, w, h) = convexHull
                # (x, y, w, h) = cv2.boundingRect(convexHull)
            # #moments = cv2.moments(erode, True)               
            # x, y, w, h = cv2.boundingRect(cnt)
            # moments = cv2.moments(cnt)         
       
        #fgmask = object_detector.apply(image_area)

        #contours = contours[0] if len(contours) == 2 else contours[1]
        #for cnt in contours:
            #cv2.waitKey()
        # print(cv2.contourArea(cnt)) #= cv2.boundingRect(cnt)
            #if cv2.contourArea(cnt) > 250:
        
                #checkStatus(estatus)          
                # x, y, w, h = cv2.boundingRect(cnt)
                # moments = cv2.moments(cnt)               
                # #moments = cv2.moments(cnt, True)               
            # area = moments['m00']
            # #ancho = x2 - x1
            #deteccions.append([x1, y1, ancho, 2])
            # #deteccions.append(lines[0])
            # deteccions.append([x, y, w, h])
            # #deteccions.append([M])
            # #deteccions.append([extLeft[0], extLeft[1], w, h])
            # boxers_ids = tracker.update(deteccions)
    
            # for box_id in boxers_ids:
            #     x, y, w, h, id = box_id
            #     cv2.putText(frame,  str(id) , (int(x), int(y)),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            #     counter = id
            
            
           # rect = cv2.minAreaRect(cnt)
            # centerX = rect[0][0]
            # centerY = rect[0][1]

        
        #     cv2.drawContours(frame, [area_pts], -1, (255,0,0), 2)
        #     cv2.putText(frame, "white: " + str(whitePixelCount), (120,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #     cv2.rectangle(frame, (400,150), (410,160), (255,0,0), 2)

		
			#cv2.putText(frame, "white: " + str(whitePixelCount), (120,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			# cv2.rectangle(frame, (420,150), (430,160), (0,255,0), 2)
			#if bandStart == True:
			#	print("Start")
				#timeON = datetime.now()
				
				# Start = datetime.now()
				
				# objeto = frame[0:400,10:800]
				# objeto = imutils.resize(objeto,width=400)
				# cv2.imwrite(Datos+'/ON_{}.jpg'.format(counter + 1),objeto)
				# bandStart = False
				
			#else:
					
			
		# else:
		# 	#cv2.drawContours(frame, [area_pts], -1, (0,0,255), 2)
		# 	cv2.putText(frame, "white: " + str(whitePixelCount), (120,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
		if bandCaptura==True:
			#timeOFF = datetime.now()
			delay = (timestampOFF - timestampON)
			OFF_formatted = timestampOFF.strftime("%H:%M:%S")
			ON_formatted = timestampON.strftime("%H:%M:%S")
			#ON_formatted = timestampOFF.strftime("%Y_%m_%d_%H_%M_%S")
			ON_formatted_image = timestampON.strftime("%Y_%m_%d_%H_%M_%S")
			#delay_formatted = format(delay,"%S")
			#ON_formatted = timestampON.strftime("%H:%M:%S")
			diff = (f-fpLast)
	#now = datetime.now()	
			#print (delay_formatted)
			
			#OFF_formatted = timestampOFF.strftime("%H:%M:%S")
	#delay = (timestampOFF - timestampON)
			# cv2.putText(frame,  str(delay.seconds)+ "."+ str(round(delay.microseconds,2)), (400,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			# cv2.putText(frGray,'#: %r' %counter, (400,200), font, 1, (0, 255, 0), 2)
			cv2.putText(frame, str("fps: ") + str(diff) , (350,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			cv2.putText(frame,  str("ON:") + str(ON_formatted), (350,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv2.putText(frame, str("OFF:") + str(OFF_formatted), (350,140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			cv2.putText(frame, str("DLY: ") + str(delay.seconds)+ "_" + str(delay.microseconds) , (350,170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			#cv2.putText(frGray, str("DELAY: ") + str(delay.seconds) , (350,170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			# [ y1:y2 , x1:x2 ]
			objeto = frame[40:480,150:600]
			objeto = imutils.resize(objeto,width=400)
			#cv2.imwrite(Datos+'/{}.jpg'.format(OFF_formattedImage + "_" + str(delay.seconds)),objeto)
			cv2.imwrite(Datos+'/{}.jpg'.format(ON_formatted_image + "_" + str(delay.seconds)+ "_" + str(delay.microseconds)),objeto)
			bandCaptura = False
			bandON = False
			fpLast=f
   
			#timeON = datetime.now()
			#ON_formatted = timeON.strftime("%H:%M:%S")
   
			# timeON = datetime.now()
			# #ON_formatted = timeON.strftime("%H:%M:%S")
			# ON_formattedImage = timeON.strftime("%Y_%m_%d_%H_%M_%S") 
   
			# objeto = frame[10:400,200:600]
			# objeto = imutils.resize(objeto,width=400)
			# #cv2.imwrite(Datos+'/{}.jpg'.format(OFF_formattedImage + "_" + str(delay.seconds)),objeto)
			# cv2.imwrite(Datos+'/{}.jpg'.format(ON_formattedImage + "_-1" ),objeto)
      
		#if bandVerde == True:
			
   
		if bandAzul == True:
			bandAzul = False
			
   		
		if bandVerde == True:
			cv2.drawContours(frame, [area_pts], -1, (0,255,0), 2)
		else:
			cv2.drawContours(frame, [area_pts], -1, (0,0,255), 2)
   
		#cv2.putText(frame, str(bandVerde), (250,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
		cv2.putText(frame,'Counter: %r' %counter, (400,30), font, 1, (0, 255, 0), 2)
       # cv2.putText(frame, "contour: " + str(len(contours)), (80,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(frame, "fp: " + str(f), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		cv2.putText(frame,  "j: " +str(j), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		cv2.putText(frame, "k: " + str(k), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		cv2.putText(frame, "min_area: " + str(minArea), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		#cv2.putText(frame,'Counter: %r' %counter, (400,120), font, 1, (0, 255, 0), 1)
        
         
		# if band == True:
		# 	cv2.rectangle(frame, (280,50), (330,100), (0,255,0), 3)
		# else:
		# 	cv2.rectangle(frame, (280,50), (330,100), (0,0,255), 3)
      
		#cv2.imshow("frGray", frGray)
			#cv2.waitKey()
		#cv2.putText(frame, "Band: " + str(band), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		#cv2.drawContours(frame, [area_pts], -1, (0,255,255), 2)
		#cv2.putText(frame, "black: " + str(f), (300,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		#cv2.putText(frame, "Value: " + str(value), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
		cv2.imshow("frame", frame)

		#cv2.imshow("aux", aux)
		cv2.imshow("result", result)
			
		cv2.imshow("background sub", fgmask)
        
       # cv2.waitKey()
		# cv2.imshow("Track", frame)
		key = cv2.waitKey(100)
		if key is ord('q'):
			break
        
#return estatus
    
Leer()
 
