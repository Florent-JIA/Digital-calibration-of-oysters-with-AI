from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
import cv2


class RemoveBackground:
    """
    RemoveBackground for removing background from the original image
    """

    def __init__(self, inputroot1, inputroot2):
        """
        :param inputroot: root for the input of the original image A(Vertical view)
        :param inputroot: root for the input of the original image B(Side view)
        """
        self.inputroot1 = inputroot1
        self.inputroot2 = inputroot2

    def removeBackground(self):
        # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with open(self.inputroot1, "rb") as input1:
            f = input1.read()

        with open(self.inputroot2, "rb") as input2:
            g = input2.read()

        result1 = remove(f)
        result2 = remove(g)
        img1 = Image.open(io.BytesIO(result1)).convert("RGBA")
        img2 = Image.open(io.BytesIO(result2)).convert("RGBA")
        return img1, img2

    def backgroundToWhite(self):
        img1, img2 = self.removeBackground()
        x1, y1 = img1.size
        p1 = Image.new('RGBA', img1.size, (255, 255, 255))
        p1.paste(img1, (0, 0, x1, y1), img1)

        x2, y2 = img2.size
        p2 = Image.new('RGBA', img2.size, (255, 255, 255))
        p2.paste(img2, (0, 0, x2, y2), img2)

        return p1, p2

class Calculation():
    def __init__(self, input1, input2, realsize1, realsize2):
        self.input1 = input1
        self.input2 = input2
        self.realsize1 = realsize1
        self.realsize2 = realsize2

    def GetCharacter(self):
        img1 = np.array(self.input1)
        img1 = 255 - img1

        img2 = np.array(self.input2)
        img2 = 255 - img2

        ret1, thresh1 = cv2.threshold(cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)

        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c1 in contours1:
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

            # draw the outlines
            if abs(bb1 - aa1) > 300:
                lg1 = np.int0(((box1[1, 0] - box1[0, 0]) ** 2 + (box1[1, 1] - box1[0, 1]) ** 2) ** 0.5)
                wt1 = np.int0(((box1[1, 0] - box1[2, 0]) ** 2 + (box1[1, 1] - box1[2, 1]) ** 2) ** 0.5)
                if (wt1 > lg1):
                    lg1, wt1 = wt1, lg1
                face1 = lg1 * wt1

                # Calculates the center and radius of the smallest closed circle
                (x1, y1), radius1 = cv2.minEnclosingCircle(c1)
                # Change into int
                center1 = (int(x1), int(y1))
                radius1 = int(radius1)

                # draw the enveloping circle in green
                if radius1 > 100:
                    img1 = cv2.circle(img1, center1, radius1, (0, 255, 0), 2)

        for c2 in contours2:
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            if w2 > 200:
                #     cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
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
            # draw the outlines
            if abs(bb2 - aa2) > 50:
                # print(box)
                lg2 = np.int0(((box2[1, 0] - box2[0, 0]) ** 2 + (box2[1, 1] - box2[0, 1]) ** 2) ** 0.5)
                wt2 = np.int0(((box2[1, 0] - box2[2, 0]) ** 2 + (box2[1, 1] - box2[2, 1]) ** 2) ** 0.5)
                if (wt2 > lg2):
                    lg2, wt2 = wt2, lg2
                face2 = lg2 * wt2
                # print("face"+","+str(face))
                cv2.drawContours(img2, [box2], 0, (0, 0, 255), 3)

            # Calculates the center and radius of the smallest closed circle
            (x2, y2), radius2 = cv2.minEnclosingCircle(c2)
            # Change into int
            center2 = (int(x2), int(y2))
            radius2 = int(radius2)
            # draw circles
            if radius2 > 100:
                img2 = cv2.circle(img2, center2, radius2, (0, 255, 0), 2)

        L = lg1 * self.realsize1
        W = wt1 * self.realsize2
        H = hight * self.realsize2
        SpaceC = (face2 - area2) * self.realsize2 * self.realsize2
        SpaceD = (face1 - area1) * self.realsize1 * self.realsize1

        return L, W, H, SpaceC, SpaceD

    def GetVariance(self):
        VA = 0
        VB = 0

        img_array1 = np.array(self.input1)
        if len(img_array1.shape) == 2:
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_GRAY2BGR)
        img1 = cv2.cvtColor(img_array1, cv2.COLOR_RGB2BGR)
        img1 = 255 - img1

        img_array2 = np.array(self.input2)
        if len(img_array2.shape) == 2:
            img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img_array2, cv2.COLOR_RGB2BGR)
        img2 = 255 - img2

        ret1, thresh1 = cv2.threshold(cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)

        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_array1 = np.array(self.input1)
        if len(img_array1.shape) == 3:
            img_gray1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
        else:
            img_gray1 = img_array1
        img1 = img_gray1

        img_array2 = np.array(self.input2)
        if len(img_array2.shape) == 3:
            img_gray2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)
        else:
            img_gray2 = img_array2
        img2 = img_gray2

        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)
        fshift1 = np.fft.fftshift(f1)
        fshift2 = np.fft.fftshift(f2)

        rows1, cols1 = img1.shape
        crow1, ccol1 = int(rows1 / 2), int(cols1 / 2)
        fshift1[crow1 - 30:crow1 + 30, ccol1 - 30:ccol1 + 30] = 0
        ishift1 = np.fft.ifftshift(fshift1)
        iimg1 = np.fft.ifft2(ishift1)
        iimg1 = np.abs(iimg1)
        height1 = iimg1.shape[0]
        weight1 = iimg1.shape[1]

        rows2, cols2 = img2.shape
        crow2, ccol2 = int(rows2 / 2), int(cols2 / 2)
        fshift2[crow2 - 30:crow2 + 30, ccol2 - 30:ccol2 + 30] = 0
        ishift2 = np.fft.ifftshift(fshift2)
        iimg2 = np.fft.ifft2(ishift2)
        iimg2 = np.abs(iimg2)
        height2 = iimg2.shape[0]
        weight2 = iimg2.shape[1]

        q1 = 0
        p1 = 0
        m1 = 0
        for row1 in range(height1):
            for col1 in range(weight1):
                dis1 = cv2.pointPolygonTest(contours1[0], (col1, row1), False)
                if int(dis1) >= 0:
                    p1 = p1 + iimg1[row1, col1]
                    m1 = m1 + (iimg1[row1, col1]) * (iimg1[row1, col1])
                    q1 = q1 + 1

        q2 = 0
        p2 = 0
        m2 = 0
        for row2 in range(height2):
            for col2 in range(weight2):
                dis2 = cv2.pointPolygonTest(contours2[0], (col2, row2), False)
                if int(dis2) >= 0:
                    p2 = p2 + iimg2[row2, col2]
                    m2 = m2 + (iimg2[row2, col2]) * (iimg2[row2, col2])
                    q2 = q2 + 1
        VA = int(m2/q2 - (p2/q2)*(p2/q2))
        VB = int(m1 / q1 - (p1 / q1) * (p1 / q1))

        return VA, VB


if __name__ == '__main__':
    inputroot1 = r'001c.png'
    inputroot2 = r'001d.png'

    RBG = RemoveBackground(inputroot1, inputroot2)

    p1, p2 = RBG.backgroundToWhite()

    Cal1 = Calculation(p2, p1, 1, 1)

    L, W, H, SpaceC, SpaceD = Cal1.GetCharacter()
    VA, VB = Cal1.GetVariance()

    print(L)
    print(W)
    print(H)
    print(SpaceC)
    print(SpaceD)
    print(VA)
    print(VB)