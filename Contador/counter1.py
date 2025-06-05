import cv2

img=cv2.imread('images/area4.jpg')
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv_img=cv2.bitwise_not(gray)
#res,thresh=cv2.threshold(inv_img,202,255,cv2.THRESH_BINARY) 
#cv2.imwrite('image2.jpg',thresh_img)

def nothing(pos):
    pass


cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',20,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

#cv2.imwrite('image1.jpg',inv_img)
while (True):
    lc = cv2.getTrackbarPos('lc','Thresholds')
    hc = cv2.getTrackbarPos('hc','Thresholds')
    
    # #edges = cv2.Canny(gray, lc, hc)
    # fgmask = cv2.dilate(thresh, None, iterations=4)
    # fgmask = cv2.erode(thresh, None, iterations=1)
    
    res,thresh=cv2.threshold(inv_img,lc, hc ,cv2.THRESH_BINARY) 
    thresh=255 - thresh
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 10:
            (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
            cv2.circle(img, (x,y),5, (0, 0, 255), 1)
        
    #   if area>4:
    #       print (area)
    #       sum1+=1
    #print sum1
    
 #print("coins : ", len(cnt))
    img = cv2.resize(img,(700,700))
    thresh = cv2.resize(thresh,(700,700))
    
    cv2.imshow('thresh', thresh) 
    cv2.imshow('img', img) 
    cv2.waitKey() 
# while (True):
#     #image = cv2.imread("images/lateral1.jpg")
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#     blur = cv2.GaussianBlur(gray, (11, 11), 0)
#     canny = cv2.Canny(blur, lc, hc, 3)
#     #dilated = cv2.dilate(canny, (1, 1), iterations=0)

    # #(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # (cnt, hierarchy) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
 
#cv2.destroyAllWindows()

    #plt.hist(areas, bins=100)
    #plt.show()
#cv2.destroyAllWindows()