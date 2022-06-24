from collections.abc import Callable
import cv2
import numpy as np
import numpy.typing as npt
from .interface import ModelInterface, ImageTransformationInterface

class CaffeDetectorImageTransformation(ModelInterface, ImageTransformationInterface):    
    """It Loads pretrained Caffe Detector models and implements image transformation interface to identify objects in the image.

    Args:
        prototxt (str): Path to .prototxt file containing model architecture.
        model_path (str): Path to .caffemodel file containing model's weight.
        model_loader (Callable, optional): Function to load the model using prototxt and model_path files. Defaults to cv2.dnn.readNetFromCaffe.
        model_preprocess (Callable, optional): Function to preprocess images to input model. Defaults to cv2.dnn.blobFromImage.
    """                
    def __init__(self, 
                prototxt: str, 
                model_path: str, 
                model_loader: Callable=cv2.dnn.readNetFromCaffe, 
                model_preprocess: Callable=cv2.dnn.blobFromImage):
        self.prototxt = prototxt
        self.model_path = model_path
        self.__model_loader = model_loader
        self.__preprocess_func = model_preprocess
        self.load_model()


    def preprocess(self, image: npt.ArrayLike, size: tuple=(300,300)) -> npt.ArrayLike:
        """Preprocesses image to loaded model

        Args:
            image (npt.ArrayLike): Raw image.
            size (tuple, optional): Height and Width to resize the raw image. Defaults to (300,300).

        Returns:
            npt.ArrayLike: Processed image.
        """        
        prep_image = cv2.resize(image, size)
        prep_image = self.__preprocess_func(prep_image, 1.0, size, (104.0, 177.0, 123.0))
        return prep_image
    
    def predict(self, image: npt.ArrayLike, size: tuple=(300,300)) -> npt.ArrayLike:
        """Run image through loaded Caffe Object Detector.

        Args:
            image (npt.ArrayLike): Raw input image.
            size (tuple, optional): Height and Width to resize the raw image. Defaults to (300,300).

        Returns:
            npt.ArrayLike: Load detector's output
        """        
        prep_image = self.preprocess(image, size)
        self.__model.setInput(prep_image)
        return self.__model.forward()
        
    def load_model(self):
            self.__model = self.__model_loader(self.prototxt, self.model_path)

    def __call__(self, image: npt.ArrayLike, confidence: float) -> list[npt.ArrayLike]:
        """Abstracts whole prediction pipeline to transform input image to output image with objects detected.

        Args:
            image (npt.ArrayLike): Input image.
            confidence (float): Considered model's confidence in detection.

        Returns:
            list[npt.ArrayLike]: List containing the image with all detections.
        """        
        h, w, _ = image.shape
        detections = self.predict(image)
        valid_detections = detections[0, 0, np.where(detections[0, 0, :, 2] > confidence)].reshape((-1, 7))
        
        for detection_information in valid_detections[:, -5:]:
            detection_confidence = detection_information[0]
            
            bbox_coordinates = detection_information[-4:]
            absolute_bbox = bbox_coordinates * np.asarray([w, h, w, h])
            x0, y0, x1, y1 = absolute_bbox.astype("int")
            
            text = f"{detection_confidence:.2f}%"
            y = y0 - 10 if y0 - 10 > 10 else y0 + 10
            
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
        return [image]