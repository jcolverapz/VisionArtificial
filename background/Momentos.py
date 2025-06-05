m = cv2.moments(cnt) # calculate x,y coordinate of center
			drawlines(frame, contours)if m["m00"] != 0:
        cX = int(m["m10"] /  m["m00"]) 
        cY = int(m["m01"] / m["m00"])
        cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
            #cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
