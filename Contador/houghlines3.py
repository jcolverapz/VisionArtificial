#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import os

def getEdges(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 110, apertureSize = 3, L2gradient = True)

def getHoughLines(image):
    minLineLength = 100
    maxLineGap = 15
    
    edges = getEdges(image)
    return cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
def getHoughLinesMask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    houghlines = getHoughLines(image)
    
    for x1,y1,x2,y2 in houghlines[0]:
        cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),4)
    
    return mask

def getStraightEdgeMask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    horizontal_lines = {}
    vertical_lines = {}
    x_lines = []
    y_lines = []
    
    houghlines = getHoughLines(image)
    
    for x1,y1,x2,y2 in houghlines[0]:
        if (x1 == x2):
            try:
                horizontal_lines[x1] = horizontal_lines[x1] + abs(y1-y2)
            except:
                horizontal_lines[x1] = abs(y1-y2)
        
        elif (y1 == y2):
            try:
                vertical_lines[y1] = vertical_lines[y1] + abs(x1-x2)
            except:
                vertical_lines[y1] = abs(x1-x2)
    
    for line in horizontal_lines:
        if horizontal_lines[line] > image.shape[0]/15:
            x_lines.append(line)
    
    for line in vertical_lines:
        if vertical_lines[line] > image.shape[1]/15:
            y_lines.append(line)
    
    x_lines.sort()
    y_lines.sort()
    print ("x-lines: %s" % x_lines)
    print ("y-lines: %s" % y_lines)
    
    for i in range(0,len(x_lines),2):
        try:
            mask[:,x_lines[i]:x_lines[i+1]] = 255
        except:
            pass
    
    for j in range(0,len(y_lines),2):
        try:
            mask[y_lines[j]:y_lines[j+1],:] = 255
        except:
            pass
    
    return cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Repair damaged image')
    parser.add_argument('-i', '--input', help='Input image file', required=True)
    parser.add_argument('-o', '--output', help='Output image file', default="output.jpg")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input) == False:
        parser.print_usage()
        exit("%s: error: input file invalid" % os.path.basename(__file__))
    
    image = cv2.imread('images/sudoku.png')
    print("Processing %s" % args.input)
    
    # Uncomment these to aid troubleshooting
    cv2.imwrite("edges.jpg",getEdges(image))
    #cv2.imwrite("hlines.jpg",getHoughLinesMask(image))
    #cv2.imwrite("4_mask.jpg",getStraightEdgeMask(image))
    
    result = cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)
    cv2.imwrite(args.output, result)