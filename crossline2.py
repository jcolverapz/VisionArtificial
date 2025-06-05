import cv2
import sys

previous_point = None  # default value at start

def click_event(event, x, y, flags, params):
    """Display the coordinates of the points clicked on the image"""
    
    global previous_point
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # left mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        text = f"{x}, {y}"

        print(text)

        if previous_point:  # check if there is previous point
            x2, y2 = previous_point
            dist = ((x-x2)**2 + (y-y2)**2 )**0.5
            #print('distance:', dist)
            print(f'distance from {x2}, {y2} to {x}, {y}:', dist)
            
            cv2.line(img, (x, y), (x2, y2), (0, 0, 0), 2)
            
            # redraw previous point to hide beginning of line 
            cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1) 
    
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, text, (x, y), font, 1, (255, 0, 0), 2)
        
        cv2.imshow('images/test9.png', img)

        previous_point = (x, y)

    # right mouse
    elif event == cv2.EVENT_RBUTTONDOWN:
        b,g,r = img[y, x, :]

        text = f"{b},{g},{r}"
        
        print(text)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)

        cv2.imshow('image', img)


# driver function

if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'images/test9.png'
        #filename = 'short.jpg'
        
    img = cv2.imread(filename, 1)
    #img = cv2.resize(img, (0, 0), None, 0.2, 0.2)

    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()