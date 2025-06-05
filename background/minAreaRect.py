# rect = cv2.minAreaRect(cnt)
			# centerX = rect[0][0]
			# box = cv2.boxPoints(rect)
			# box = np.int_(box)
			# #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
			# #cv2.drawContours(frame,box,0,(0,255,0),1)
   
   
   rect = cv2.minAreaRect(cnt)
			centerX = rect[0][0]
			box = cv2.boxPoints(rect)
			box = np.int_(box)
   