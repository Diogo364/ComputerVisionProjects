from collections.abc import Callable
import cv2
import dlib
import numpy as np
import numpy.typing as npt
from .interface import ModelInterface, ImageTransformationInterface

class DlibLandmarkDetectorImageTransformation(ModelInterface, ImageTransformationInterface):    
    """It Loads pretrained Dlib Face Landmark model and implements image transformation interface to locate face landmarks in the image.

    Args:
        model_path (str): Path to .dat file containing model's weight.
        model_loader (Callable, optional): Function to load the model using prototxt and model_path files. Defaults to cv2.dnn.readNetFromCaffe.
        model_preprocess (Callable, optional): Function to preprocess images to input model. Defaults to cv2.dnn.blobFromImage.
    """                
    def __init__(self, 
                model_path: str, 
                model_loader: Callable=dlib.shape_predictor, 
                model_preprocess: Callable=dlib.get_frontal_face_detector()):
        self.model_path = model_path
        self.__model_loader = model_loader
        self.__preprocess_func = model_preprocess
        self.load_model()


    @staticmethod
    def _get_point_coordinates(pt: dlib.point) -> tuple[int]:
        """Extract (x, y) coordinates from point object.

        Args:
            pt (dlib.point): Point object

        Returns:
            tuple[int]: x and y coordinates.
        """        
        return pt.x, pt.y
    
    def _get_rect_coordinates(self, rect: dlib.rectangle) -> tuple:
        """Extract tl and br coordinates from rectangle object.

        Args:
            rect (dlib.rectangle): Rectangle object.

        Returns:
            tuple: tl and br coordinates.
        """        
        tl_corner = rect.tl_corner()
        br_corner = rect.br_corner()
        return self._get_point_coordinates(tl_corner), self._get_point_coordinates(br_corner)
    
    def preprocess(self, image: npt.ArrayLike, num_upsamples: int=1) -> tuple[npt.ArrayLike, dlib.rectangles]:
        """Preprocesses image to loaded model detecting all faces within the image.

        Args:
            image (npt.ArrayLike): Raw image.
            num_upsamples (int, optional): Number of Upsample processes over image. Defaults to 1.

        Returns:
            tuple[npt.ArrayLike, dlib.rectangles]: Tuple containing the processed image and Rectangles containing face coordinates
        """        
        prep_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        recs = self.__preprocess_func(prep_image, num_upsamples)
        return prep_image, recs
    
    def predict(self, image: npt.ArrayLike, num_upsamples: int=1) -> npt.ArrayLike:
        """Run image through loaded Face Landmarks Detector.

        Args:
            image (npt.ArrayLike): Raw input image.
            num_upsamples (int, optional): Number of Upsample processes over image. Defaults to 1.


        Returns:
            npt.ArrayLike: Load detector's output
        """        
        processed_img, recs = self.preprocess(image, num_upsamples)
        for idx, rec in enumerate(recs):
            shape = self.__model(processed_img, rec)
            shape_array = np.asarray([self._get_point_coordinates(pt) for pt in shape.parts()])
            
            for x, y in shape_array:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        return image
        
    def load_model(self):
            self.__model = self.__model_loader(self.model_path)

    def __call__(self, image: npt.ArrayLike, num_upsamples: int=1) -> list[npt.ArrayLike]:
        """Abstracts whole prediction pipeline to transform input image to output image with objects detected.

        Args:
            image (npt.ArrayLike): Input image.
            num_upsamples (int, optional): Number of Upsample processes over image. Defaults to 1.

        Returns:
            list[npt.ArrayLike]: List containing the image with all detections.
        """        
        image = self.predict(image, num_upsamples)
            
        return [image]