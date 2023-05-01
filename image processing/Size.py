from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import cmath
# read images

input_path = 'C:/Users/Tardis/Documents/PJENT-qualite-huitre/data_white_background'
'''for root, dirs, files in os.walk(input_path):
    for name in files:
        file = os.path.join(root, name)
        print('transform' + name)
        im = Image.open(file)
        x, y = im.size
        p = Image.new('RGBA', im.size, (255, 255, 255))
        p.paste(im, (0, 0, x, y), im)
        p.save('green'+name)'''
for root, dirs, files in os.walk(input_path):
    for name in files:
        file = os.path.join(root, name)
        #print('transform' + name)
       # img = Image.open(file)
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = 255 - img
        #print(img.shape)

        #p = Image.new('RGBA', img.size, (255, 255, 255))
        #p.paste(img, (0, 0, x, y), img)
        #img = p
        #img= cv2.medianBlur(img, 5)
       # img = cv2.blur(img,(5,5))

        # Binary
        ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),20, 255,cv2.THRESH_BINARY)
        # #cv2.imshow("contours", thresh)

        # Search boundry
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        a = "d" in name
        if a == True:
            for c in contours:

                x, y, w, h = cv2.boundingRect(c)
                if w>50:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Get the smallest rectangular profile May have a rotation angle
                rect = cv2.minAreaRect(c)
                # Calculates the coordinates of the smallest area
                box = cv2.boxPoints(rect)
                # nomalize the coordinates into int
                box = np.int0(box)

                bb = box[0,0]
                aa = box[2,0]

                area = 0

                for i in contours:
                    area += cv2.contourArea(i)
                #print("area"+","+str(area))
                # draw the outlines
                if (aa - bb)>300:
                   # print(box)
                    lg = np.int0(((box[1,0]-box[0,0])**2+(box[1,1]-box[0,1])**2)**0.5)
                    wt = np.int0(((box[1,0]-box[2,0])**2+(box[1,1]-box[2,1])**2)**0.5)
                    if (wt > lg):
                        lg,wt=wt,lg
                    #print(name+","+"length"+","+str(lg))
                    #print(name+","+"width"+","+str(wt))
                    face = lg*wt
                    #print("face"+","+str(face))
                    print(name+","+"d"+","+"face - area"+","+str(face - area))
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)


                # Calculates the center and radius of the smallest closed circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # Change into int
                center = (int(x), int(y))
                radius = int(radius)
                #print(radius)
                # draw circles
                if radius>100:
                    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

            # draw outlines
            cv2.namedWindow("contours", 0);
            cv2.resizeWindow("contours", 1920, 1080);
            cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
            cv2.imshow("contours", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            for c in contours:

                x, y, w, h = cv2.boundingRect(c)
                if w > 200:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    hight = h
                    #print(name+","+"hight"+","+str(hight))
                    # Get the smallest rectangular profile May have a rotation angle
                rect = cv2.minAreaRect(c)
                # Calculates the coordinates of the smallest area
                box = cv2.boxPoints(rect)
                # nomalize the coordinates into int
                box = np.int0(box)

                bb = box[0, 0]
                aa = box[2, 0]

                area = 0
                for i in contours:
                    area += cv2.contourArea(i)
                #print("area"+","+str(area))

                # draw the outlines
                if (aa - bb) > 300:
                    #print(box)
                    lg = np.int0(((box[1, 0] - box[0, 0]) ** 2 + (box[1, 1] - box[0, 1]) ** 2) ** 0.5)
                    wt = np.int0(((box[1, 0] - box[2, 0]) ** 2 + (box[1, 1] - box[2, 1]) ** 2) ** 0.5)
                    if (wt > lg):
                        lg, wt = wt, lg
                    face = lg*wt
                    print(name+","+"c"+","+"face - area"+","+str(face - area))
                    #print(lg)
                    #print(wt)
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

                # Calculates the center and radius of the smallest closed circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # Change into int
                center = (int(x), int(y))
                radius = int(radius)
                # print(radius)
                # draw circles
                if radius > 100:
                    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

                # draw outlines
            cv2.namedWindow("contours", 0);
            cv2.resizeWindow("contours", 1920, 1080);
            cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
            cv2.imshow("contours", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
