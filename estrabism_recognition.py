# Python version 3.8.2
# Numpy library instalation on linux terminal:  sudo pip3 install numpy
import numpy as np

# Opencv2 library instalation on linux terminal: sudo pip3 install opencv-python
import cv2

# Function for irish selection with differences with centroid
def select_irish(circles, eye_color, y, x):
    dark_circle = circles[0][0]
    if(len(circles[0]) > 1):
        for i in range(1, len(circles[0])): 
            a, b, r = circles[0][i]
            a2, b2, r2 = dark_circle
            if( (abs(y-b) < abs(y-b2) ) and (abs(x-a) < abs(x-a2) ) ):
                dark_circle = circles[0][i]
    return dark_circle


#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('eyeClass.xml')

img = cv2.imread('./faces/face5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in eyes:
    eye = gray[y:y+h, x:x+w]
    eye_color = img[y:y+h, x:x+w]
    
    h2, w2 = eye.shape
    center_y = int(h2/2)
    center_x = int(w2/2)
 
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(eye, (3, 3))
  
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 10, maxRadius = 40) 

    level1x = int(w2/15)
    level2x = int(w2/8)
    level3x = int(w2/3)

    level1y = int(h2/15)
    level2y = int(h2/8)
    level3y = int(h2/3)
  
    # Draw circles that are detected.
    if detected_circles is not None: 
  
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles))

        pt = select_irish(detected_circles, eye, center_y, center_x)

        a, b, r = pt[0], pt[1], pt[2]
 
        # Draw the circumference of the circle. 
        cv2.circle(eye_color, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(eye_color, (a, b), 1, (0, 0, 255), 3)

        eye_xcolor = eye_color.copy()
        eye_ycolor = eye_color.copy()

        # Horizontal analysis of eye
        eye_xcolor = cv2.rectangle(eye_xcolor, (center_x-level1x, 0), (center_x+level1x ,h2), (255, 0, 0), 1)
        eye_xcolor = cv2.rectangle(eye_xcolor, (center_x-level2x, 0), (center_x+level2x ,h2), (0, 255, 255), 1)
        eye_xcolor = cv2.rectangle(eye_xcolor, (center_x-level3x, 0), (center_x+level3x ,h2), (0, 0, 255), 1)
        print("\n Análisis Horizontal: ")
        if(a >= center_x-level1x and a <= center_x+level1x):
            print( 'Parámetros normales' )
        elif(a >= center_x-level2x and a <= center_x+level2x):
            print( 'Estrabismo Horizontal Leve' )
        else:
            print( 'Estrabismo Horizontal Grave' )   

        # Vertical analysis of eye
        eye_ycolor = cv2.rectangle(eye_ycolor, (0, center_y-level1y), (w2 , center_y+level1y), (255, 0, 0), 1)
        eye_ycolor = cv2.rectangle(eye_ycolor, (0, center_y-level2y), (w2 , center_y+level2y), (0, 255, 255), 1)
        eye_ycolor = cv2.rectangle(eye_ycolor, (0, center_y-level3y), (w2 , center_y+level3y), (0, 0, 255), 1)

        print( "\n Análisis Vertical: " )
        if(b >= center_y-level1y and b <= center_y+level1y):
            print( 'Parámetros normales' )
        elif(b >= center_y-level2y and b <= center_y+level2y):
            print( 'Estrabismo Horizontal Leve' )
        else:
            print( 'Estrabismo Horizontal Grave' )

        cv2.imshow("Ojo Detectado", eye_color)
        cv2.imshow("Horizontal", eye_xcolor)
        cv2.imshow("Vertical", eye_ycolor)
        cv2.waitKey(0)
        