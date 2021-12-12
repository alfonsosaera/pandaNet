import cv2

class Resizing:
    """image resizing preprocessor"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preproccess(self, img):
        # resize the image
        return cv2.resize(img, (self.width, self.height))
