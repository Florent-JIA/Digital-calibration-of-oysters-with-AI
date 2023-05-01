import cv2
import numpy as np


class GetVariance:
    """
    GetVariance for getting Variance from the data
    """
    input_path1 = "outA.png"
    input_path2 = "outB.png"
    VA = 0
    VB = 0
    def __init__(self):
        img1 = cv2.imread(self.input_path1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(self.input_path2, cv2.IMREAD_UNCHANGED)
        img1 = 255 - img1
        img2 = 255 - img2


        ret1, thresh1 = cv2.threshold(cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        # #cv2.imshow("contours", thresh)

        # Search boundry
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # draw outlines
        #cv2.namedWindow("contours", 0);
        #cv2.resizeWindow("contours", 1920, 1080);
        cv2.drawContours(img1, contours1, -1, (255, 0, 0), 1)
        cv2.drawContours(img2, contours2, -1, (255, 0, 0), 1)
        #cv2.imshow("contours", img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        #img = cv2.imread("C:/Users/Tardis/Desktop/huit/data_green/green001c.png", 0)
        img1 = cv2.imread(self.input_path1, 0)
        img2 = cv2.imread(self.input_path2, 0)
        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)
        fshift1 = np.fft.fftshift(f1)
        fshift2 = np.fft.fftshift(f2)



        rows1,cols1 = img1.shape
        crow1,ccol1 = int(rows1/2), int(cols1/2)
        fshift1[crow1-30:crow1+30, ccol1-30:ccol1+30] = 0
        ishift1 = np.fft.ifftshift(fshift1)
        iimg1 = np.fft.ifft2(ishift1)
        iimg1 = np.abs(iimg1)
        height1 = iimg1.shape[0]
        weight1 = iimg1.shape[1]

        rows2,cols2 = img2.shape
        crow2,ccol2 = int(rows2/2), int(cols2/2)
        fshift2[crow2-30:crow2+30, ccol2-30:ccol2+30] = 0
        ishift2 = np.fft.ifftshift(fshift2)
        iimg2 = np.fft.ifft2(ishift2)
        iimg2 = np.abs(iimg2)
        height2 = iimg2.shape[0]
        weight2 = iimg2.shape[1]


        #print(height)
        #print(weight)
        q1 = 0
        p1 = 0
        m1 = 0
        for row1 in range(height1):
            for col1 in range(weight1):
                dis1 = cv2.pointPolygonTest(contours1[0], (col1, row1), False)
                if int(dis1) >= 0:
                    p1 = p1+iimg1[row1,col1]
                    m1 = m1 + (iimg1[row1,col1])*(iimg1[row1,col1])
                    q1=q1+1

        q2 = 0
        p2 = 0
        m2 = 0
        for row2 in range(height2):
            for col2 in range(weight2):
                dis2 = cv2.pointPolygonTest(contours2[0], (col2, row2), False)
                if int(dis2) >= 0:
                    p2 = p2+iimg2[row2,col2]
                    m2 = m2 + (iimg2[row2,col2])*(iimg2[row2,col2])
                    q2=q2+1

        self.VA = int(m2/q2 - (p2/q2)*(p2/q2))
        self.VB = int(m1 / q1 - (p1 / q1) * (p1 / q1))
        #print(self.input_path1+","+str(int(m1/q1 - (p1/q1)*(p1/q1))))
        #print(self.input_path2+","+str(int(m2/q2 - (p2/q2)*(p2/q2))))




def main():
    xx = GetVariance()
    print(xx.VA)
    print(xx.VB)



if __name__ == '__main__':
    main()