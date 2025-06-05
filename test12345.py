for cntr in range(0, len(contours)):
        cntr = contours[i]
        size = cv2.contourArea(cntr)
        if size < 1000:
            M = cv2.moments(cntr)
            cX = int(M["m10"] / (M["m00"] + 1e-5))
            cY = int(M["m01"] / (M["m00"] + 1e-5))
            listx.append(cX)
            listy.append(cY)

    listxy = list(zip(listx,listy))
    listxy = np.array(listxy)

    for x1, y1 in listxy:    
        distance = 0
        secondx = []
        secondy = []
        dist_listappend = []
        sort = []   
        for x2, y2 in listxy:      
            if (x1, y1) == (x2, y2):
                pass     
            else:
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                secondx.append(x2)
                secondy.append(y2)
                dist_listappend.append(distance)               
        secondxy = list(zip(dist_listappend,secondx,secondy))
        sort = sorted(secondxy, key=lambda second: second[0])
        sort = np.array(sort)
        cv2.line(img, (x1,y1), (int(sort[0,1]), int(sort[0,2])), (0,0,255), 2)

    cv2.imshow('img', img)
    cv2.imwrite('connected.png', img)