import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_path = 'C:/Users/Tardis/Documents/PJENT-qualite-huitre/data_white_background'
for root, dirs, files in os.walk(input_path):
    for name in files:
        file = os.path.join(root, name)

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = 255 - img
        ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        # 图像阈值化，灰度值20以下的点置零，20以上的点置255
        # ret是阈值，thresh是处理后的图像的数组
        # #cv2.imshow("contours", thresh)

        # Search boundry
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 找到轮廓contours

        # draw outlines
        # cv2.namedWindow("contours", 0);
        # cv2.resizeWindow("contours", 1920, 1080);
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  # 在原始图片上画轮廓
        # cv2.imshow("contours", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # img = cv2.imread("C:/Users/Tardis/Desktop/huit/data_green/green001c.png", 0)
        img = cv2.imread(file, 0)  # 重置img，以灰度模式加载图片
        f = np.fft.fft2(img)
        # 傅里叶变换，得到频谱图
        # 频谱图中频率高低表征图像中灰度变化的剧烈程度。图像中边缘和噪声往往是高频信号，而图像背景往往是低频信号
        # 在频率域内可以很方便地对图像的高频或低频信息进行操作，完成图像去噪，图像增强，图像边缘提取等操作
        fshift = np.fft.fftshift(f)
        # 由于变换完后的复数数组对应的频率在数组中的出现顺序是0(直流分量)，正频率，负频率，所以很不对称
        # 故利用函数np.fft.fftshift对得到的复数数组移位
        # 使得数组中复数对应的频率是关于0对称的
        # (即尽量让直流分量出现在数组的正中间，负频率出现在直流分量左边，正频率出现在直流分量右边，这么做的原因主要是为了画图好看)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        # 使用一个60*60的矩形窗口对图像进行掩模操作从而去除低频分量
        print(fshift)
        ishift = np.fft.ifftshift(fshift)
        # 进行逆平移操作，使直流分量又回到左上角
        iimg = np.fft.ifft2(ishift)
        # 傅里叶逆变换
        iimg = np.abs(iimg)
        height = iimg.shape[0]
        weight = iimg.shape[1]

        # print(height)
        # print(weight)
        q = 0  # 轮廓内点的个数
        p = 0  # 轮廓内点的像素值累加
        m = 0  # 轮廓内点的像素值平方的累加
        for row in range(height):  # 遍历高
            for col in range(weight):  # 遍历宽
                dis = cv2.pointPolygonTest(contours[0], (col, row), False)  # 检查点与轮廓的相对位置，1内0上-1外
                if int(dis) >= 0:
                    p = p + iimg[row, col]
                    m = m + (iimg[row, col]) * (iimg[row, col])
                    q = q + 1
        # print(q)
        # print(p)
        # print(m)
        print(name + "," + str(int(m / q - (p / q) * (p / q))))
        # 平方的累加的平均数减去累加的平均数的平方

        plt.subplot(211)

        plt.imshow(img.astype('uint8'), cmap='gray')

        plt.title('original')
        plt.axis('off')

        plt.subplot(212)

        plt.imshow(iimg.astype('uint8'), cmap='gray')

        plt.title('FFT')
        plt.axis('off')

        plt.show()
