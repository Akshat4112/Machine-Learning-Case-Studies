from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFromat = None):
        self.dataFromat = dataFromat
        
    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFromat)