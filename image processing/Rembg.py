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
    output_path1 = "outA.png"
    output_path2 = "outB.png"
    def __init__(self, inputroot1,inputroot2):
        """
        :param inputroot: root for the input of the original image A(Vertical view)
        :param inputroot: root for the input of the original image B(Side view)
        """
        self.inputroot1 = inputroot1
        self.inputroot2 = inputroot2

    def removeBackground(self):

        # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        f = np.fromfile(self.inputroot1)
        g = np.fromfile(self.inputroot2)
        result1 = remove(f)
        result2 = remove(g)
        img1 = Image.open(io.BytesIO(result1)).convert("RGBA")
        img2 = Image.open(io.BytesIO(result2)).convert("RGBA")
        img1.save(self.output_path1)
        img2.save(self.output_path2)
        return img1, img2

    def backgroundToWhite(self):
        img1 = Image.open(self.output_path1)
        x1, y1 = img1.size
        p1 = Image.new('RGBA', img1.size, (255, 255, 255))
        p1.paste(img1, (0, 0, x1, y1), img1)
        p1.save(self.output_path1)

        img2 = Image.open(self.output_path2)
        x2, y2 = img2.size
        p2 = Image.new('RGBA', img2.size, (255, 255, 255))
        p2.paste(img2, (0, 0, x2, y2), img2)
        p2.save(self.output_path2)



def main():
    input_path1 = "001d.jpg"
    input_path2 = "001c.jpg"
    xx = RemoveBackground(input_path1,input_path2)
    xx.removeBackground()
    xx.backgroundToWhite()


if __name__ == '__main__':
    main()