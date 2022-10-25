from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from .documentScannerImageTransformation import DocumentScannerImageTransformation
from .interface import ImageTransformationInterface


class BubbleExtractorImageTransformation(ImageTransformationInterface):
    """Extract Answers from a Bubble Card

    Args:
        answer_options (list, optional): Ordered (left to right) list of possible answers. Defaults to ['a', 'b', 'c', 'd', 'e'].
    """        
    def __init__(self, answer_options=['a', 'b', 'c', 'd', 'e']):
        self.__doc_scanner = DocumentScannerImageTransformation()
        self._question_cnts = []
        self.__answer_options = answer_options
        self.answers = {}
        self.__answer_contours = []

    @staticmethod
    def __sort_contours(pts: npt.ArrayLike, left_right: bool=True) -> npt.ArrayLike:
        """Sort contour list from its minimum position.

        Args:
            pts (npt.ArrayLike): Array of contours.
            left_right (bool, optional): Orders from left to right if True. Orders from top to bottom if False. Defaults to True.

        Returns:
            npt.ArrayLike: Ordered contour list.
        """        
        order_element = 0 if left_right else 1
        return sorted(pts, key=lambda x: x.min(axis=0).flatten()[order_element])
    

    def __set_question_bubbles(self, image: npt.ArrayLike):
        """Identify bubbles in the cart.
        It searches for boundingBoxes of specific minimum size and aspect ratio.

        Args:
            image (npt.ArrayLike): Grayscale image.
        """        
        cnts, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            if w >= 15 and h >= 15 and ratio >= 0.9 and ratio <= 1.5:
                self._question_cnts.append(cnt)

    @staticmethod
    def __extract_answer_bubble(image: npt.ArrayLike, bubbles: npt.ArrayLike) -> Tuple[int, int]:
        """Identify the answered bubble from a list of bubbles.

        Args:
            image (npt.ArrayLike): Binarized image.
            bubbles (npt.ArrayLike): List of bubbles.

        Returns:
            tuple[int, int]: Tuple containing the total of filled points and the answer postion in the input bubble list.
        """        
        bubbled = -np.inf, 0
        for letter_idx, bubble_cnt in enumerate(bubbles):
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [bubble_cnt], -1, 255, -1)
            mask = cv2.bitwise_and(image, image, mask=mask)
            total = cv2.countNonZero(mask)
            if total > bubbled[0]:
                bubbled = (total, letter_idx)
        return bubbled

    def __set_answers(self, image: npt.ArrayLike, n_choices: int=5):
        """Iterates over sets of answers and extract the choosen bubble for each of them.

        Args:
            image (npt.ArrayLike): Binary image.
            n_choices (int, optional): Number of possible answers. Defaults to 5.
        """        
        for question_number, contour_row in enumerate(np.arange(0, len(self._question_cnts), n_choices), start=1):
            left_right_bubbles = self.__sort_contours(self._question_cnts[contour_row:contour_row+n_choices])
            _, answer_id = self.__extract_answer_bubble(image, left_right_bubbles)
            self.answers[question_number] = self.__answer_options[answer_id]
            
            self.__answer_contours.append(left_right_bubbles[answer_id])

    
    def __call__(self, image: npt.ArrayLike) -> List[npt.ArrayLike]:
        smart_cropped_image = self.__doc_scanner(image.copy(), binarization=False)[0]
        processed_image = cv2.cvtColor(smart_cropped_image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self.__set_question_bubbles(binary_image)
        self._question_cnts = self.__sort_contours(self._question_cnts, left_right=False)
        self.__set_answers(binary_image)
        smart_cropped_image = cv2.drawContours(smart_cropped_image, self.__answer_contours, -1, (0, 255, 0), 3)
        text = f"Answers: {self.answers}"
        cv2.putText(smart_cropped_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        return [smart_cropped_image]