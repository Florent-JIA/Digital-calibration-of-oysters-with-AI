import cv2
import numpy as np
import csv
from GetVariance import GetVariance

class GetCharacter:
    """
    GetCharacter for getting Characters from the data
    """
    input_path1 = "outA.png"
    input_path2 = "outB.png"
    csv_path = "out.csv"
    def __init__(self,realSize1,realSize2):
        """
        :param realSize1: The real size of the area covered by the image A (Vertical view)
        :param realSize2: The real size of the area covered by the image B (Side view)
        """
        self.realSize1 = realSize1
        self.realSize2 = realSize2


        img1 = cv2.imread(self.input_path1, cv2.IMREAD_UNCHANGED)
        img1 = 255 - img1
        img2 = cv2.imread(self.input_path2, cv2.IMREAD_UNCHANGED)
        img2 = 255 - img2
        # Binary
        ret1, thresh1 = cv2.threshold(cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)


        # Search boundry
        contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c1 in contours1:

            x1, y1, w1, h1 = cv2.boundingRect(c1)
            if w1 > 200:
                cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2) # 用绿色画出最小矩形框

                # Get the smallest rectangular profile May have a rotation angle
            rect1 = cv2.minAreaRect(c1)
            # Calculates the coordinates of the smallest area
            box1 = cv2.boxPoints(rect1)
            # nomalize the coordinates into int
            box1 = np.int0(box1)
            bb1 = box1[0, 0]
            aa1 = box1[2, 0]
            area1 = 0

            for i1 in contours1:
                area1 += cv2.contourArea(i1)
            # print("area"+","+str(area))
            # draw the outlines
            if abs(bb1 - aa1) > 300:
                # print(box)
                lg1 = np.int0(((box1[1, 0] - box1[0, 0]) ** 2 + (box1[1, 1] - box1[0, 1]) ** 2) ** 0.5)
                wt1 = np.int0(((box1[1, 0] - box1[2, 0]) ** 2 + (box1[1, 1] - box1[2, 1]) ** 2) ** 0.5)
                if (wt1 > lg1):
                    lg1, wt1 = wt1, lg1
                # print(name+","+"length"+","+str(lg))
                # print(name+","+"width"+","+str(wt))
                face1 = lg1 * wt1
                cv2.drawContours(img1, [box1], 0, (0, 0, 255), 3) # 用蓝色画出轮廓

            # Calculates the center and radius of the smallest closed circle
            (x1, y1), radius1 = cv2.minEnclosingCircle(c1)
            # Change into int
            center1 = (int(x1), int(y1))
            radius1 = int(radius1)
            # print(radius)
            # draw circles
            if radius1 > 100:
                img1 = cv2.circle(img1, center1, radius1, (0, 255, 0), 2) #用绿色画出绿色圆框
        cv2.namedWindow("contours1", 0);
        cv2.resizeWindow("contours1", 1920, 1080);
        cv2.drawContours(img1, contours1, -1, (255, 0, 0), 1) # 用红色画出旋转矩形框
        cv2.imshow("contours1", img1)
        cv2.waitKey()
        cv2.destroyAllWindows()


        for c2 in contours2:

            x2, y2, w2, h2 = cv2.boundingRect(c2)
            if w2 > 200:
                cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                hight = h2
                # Get the smallest rectangular profile May have a rotation angle
            rect2 = cv2.minAreaRect(c2)
            # Calculates the coordinates of the smallest area
            box2 = cv2.boxPoints(rect2)
            # nomalize the coordinates into int
            box2 = np.int0(box2)
            bb2 = box2[0, 0]
            aa2 = box2[2, 0]
            area2 = 0

            for i2 in contours2:
                area2 += cv2.contourArea(i2)
            # print("area"+","+str(area))
            # draw the outlines
            if abs(bb2 - aa2) > 50:
                # print(box)
                lg2 = np.int0(((box2[1, 0] - box2[0, 0]) ** 2 + (box2[1, 1] - box2[0, 1]) ** 2) ** 0.5)
                wt2 = np.int0(((box2[1, 0] - box2[2, 0]) ** 2 + (box2[1, 1] - box2[2, 1]) ** 2) ** 0.5)
                if (wt2 > lg2):
                    lg2, wt2 = wt2, lg2
                # print(name+","+"length"+","+str(lg))
                # print(name+","+"width"+","+str(wt))
                face2 = lg2 * wt2
                # print("face"+","+str(face))
                cv2.drawContours(img2, [box2], 0, (0, 0, 255), 3)

            # Calculates the center and radius of the smallest closed circle
            (x2, y2), radius2 = cv2.minEnclosingCircle(c2)
            # Change into int
            center2 = (int(x2), int(y2))
            radius2 = int(radius2)
            # print(radius)
            # draw circles
            if radius2 > 100:
                img2 = cv2.circle(img2, center2, radius2, (0, 255, 0), 2)
        cv2.namedWindow("contours2", 0);
        cv2.resizeWindow("contours2", 1920, 1080);
        cv2.drawContours(img2, contours2, -1, (255, 0, 0), 1)
        cv2.imshow("contours2", img2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        xx = GetVariance()

        r = open(self.csv_path,'w',encoding='utf-8',newline="")
        writer = csv.writer(r)
        writer.writerow(['ID','Lable','L','W','H','SpaceC','SpaceD','VA','VB'])
        writer.writerow([1, 0, lg1*self.realSize1, wt1*self.realSize1, hight*self.realSize2, (face2 - area2)*self.realSize2*self.realSize2,(face1 - area1)*self.realSize1*self.realSize1, xx.VA, xx.VB])
        print("ID, "+"1"+", Length, "+ str(lg1*self.realSize1)+ ", Width, "+ str(wt1*self.realSize1)+ ", Hight, "+str(hight*self.realSize2)+ ", SpaceA, "+ str((face2 - area2)*self.realSize2*self.realSize2)+", SpaceB, "+ str((face1 - area1)*self.realSize1*self.realSize1)+", VA, "+str(xx.VA)+", VB, "+str(xx.VB))
        r.close()
def main():
    GetCharacter(1,1)

if __name__ == '__main__':
    main()