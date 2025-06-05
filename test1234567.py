x = approx.ravel()[0]
		y = approx.ravel()[1]

		x1 ,y1, w, h = cv2.boundingRect(approx)
		a=w*h    
		if len(approx) == 4 and x>15  :
				
			aspectRatio = float(w)/h
			if  aspectRatio >= 2.5 and a>100:          
			#print(x1,y1,w,h)
				width=w
				height=h   
				start_x=x1
				start_y=y1
				end_x=start_x+width
				end_y=start_y+height      
				cv2.rectangle(output2, (start_x,start_y), (end_x,end_y), (0,0,255),3)
				cv2.putText(output2, "rectangle "+str(x1)+" , " +str(y1-5), (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
			
	cv2.imshow("op",output)