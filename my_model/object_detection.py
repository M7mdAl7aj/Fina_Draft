
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from utilities import get_path, show_image


class ObjectDetector:
    """
        A class for detecting objects in images using models like Detic and YOLOv5.

        This class supports loading and using different object detection models to identify objects
        in images and draw bounding boxes around them.

        Attributes:
            model (torch.nn.Module): The loaded object detection model.
            processor (transformers.AutoImageProcessor): Processor for the Detic model.
            model_name (str): Name of the model used for detection.
        """

    def __init__(self):
        """
        Initializes the ObjectDetector class with default values.
        """

        self.model = None
        self.processor = None
        self.model_name = None

    def load_model(self, model_name='detic', pretrained=True, model_version='yolov5s'):
        """
                Load the specified object detection model.

                Args:
                    model_name (str): Name of the model to load. Options are 'detic' and 'yolov5'.
                    pretrained (bool): Boolean indicating if a pretrained model should be used.
                    model_version (str): Version of the YOLOv5 model, applicable only when using YOLOv5.

                Raises:
                    ValueError: If an unsupported model name is provided.
        """

        self.model_name = model_name
        if model_name == 'detic':
            self._load_detic_model(pretrained)
        elif model_name == 'yolov5':
            self._load_yolov5_model(pretrained, model_version)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def _load_detic_model(self, pretrained):
        """
        Load the Detic model.

        Args:
            pretrained (bool): If True, load a pretrained model.
        """

        try:
            model_path = get_path('deformable-detr-detic', 'models')
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading Detic model: {e}")
            raise

    def _load_yolov5_model(self, pretrained, model_version):
        """
        Load the YOLOv5 model.

        Args:
            pretrained (bool): If True, load a pretrained model.
            model_version (str): Version of the YOLOv5 model.
        """

        try:
            model_path = get_path('yolov5', 'models')
            if model_path and os.path.exists(model_path):
                self.model = torch.hub.load(model_path, model_version, pretrained=pretrained, source='local')
            else:
                self.model = torch.hub.load('ultralytics/yolov5', model_version, pretrained=pretrained)
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            raise

    def process_image(self, image_path):
        """
        Process the image from the given path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Image.Image: Processed image in RGB format.

        Raises:
            Exception: If an error occurs during image processing.
        """

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    def detect_objects(self, image, threshold=0.4):
        """
        Detect objects in the given image using the loaded model.

        Args:
            image (Image.Image): Image in which to detect objects.
            threshold (float): Model detection confidence.

        Returns:
            tuple: A tuple containing a string representation and a list of detected objects.

        Raises:
            ValueError: If the model is not loaded or the model name is unsupported.
        """

        if self.model_name == 'detic':
            return self._detect_with_detic(image, threshold)
        elif self.model_name == 'yolov5':
            return self._detect_with_yolov5(image, threshold)
        else:
            raise ValueError("Model not loaded or unsupported model name")

    def _detect_with_detic(self, image, threshold):
        """
        Detect objects using the Detic model.

        Args:
            image (Image.Image): The image in which to detect objects.
            threshold (float): The confidence threshold for detections.

        Returns:
            tuple: A tuple containing a string representation and a list of detected objects.
                   Each object in the list is represented as a tuple (label_name, box_rounded, certainty).
        """

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
            0]

        detected_objects_str = ""
        detected_objects_list = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= threshold:
                label_name = self.model.config.id2label[label.item()]
                box_rounded = [round(coord, 2) for coord in box.tolist()]
                certainty = round(score.item() * 100, 2)
                detected_objects_str += f"{{object: {label_name}, bounding box: {box_rounded}, certainty: {certainty}%}}\n"
                detected_objects_list.append((label_name, box_rounded, certainty))
        return detected_objects_str, detected_objects_list

    def _detect_with_yolov5(self, image, threshold):
        """
        Detect objects using the YOLOv5 model.

        Args:
            image (Image.Image): The image in which to detect objects.
            threshold (float): The confidence threshold for detections.

        Returns:
            tuple: A tuple containing a string representation and a list of detected objects.
                   Each object in the list is represented as a tuple (label_name, box_rounded, certainty).
        """

        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.model(cv2_img)

        detected_objects_str = ""
        detected_objects_list = []
        for *bbox, conf, cls in results.xyxy[0]:
            if conf >= threshold:
                label_name = results.names[int(cls)]
                box_rounded = [round(coord.item(), 2) for coord in bbox]
                certainty = round(conf.item() * 100, 2)
                detected_objects_str += f"{{object: {label_name}, bounding box: {box_rounded}, certainty: {certainty}%}}\n"
                detected_objects_list.append((label_name, box_rounded, certainty))
        return detected_objects_str, detected_objects_list

    def draw_boxes(self, image, detected_objects, show_confidence=True):
        """
        Draw bounding boxes around detected objects in the image.

        Args:
            image (Image.Image): Image on which to draw.
            detected_objects (list): List of detected objects.
            show_confidence (bool): Whether to show confidence scores.

        Returns:
            Image.Image: Image with drawn boxes.
        """

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        label_color_map = {}

        for label_name, box, score in detected_objects:
            if label_name not in label_color_map:
                label_color_map[label_name] = colors[len(label_color_map) % len(colors)]

            color = label_color_map[label_name]
            draw.rectangle(box, outline=color, width=3)

            label_text = f"{label_name}"
            if show_confidence:
                label_text += f" ({round(score, 2)}%)"
            draw.text((box[0], box[1]), label_text, fill=color, font=font)

        return image


def detect_and_draw_objects(image_path, model_type='yolov5', threshold=0.2, show_confidence=True):
    """
    Detects objects in an image, draws bounding boxes around them, and returns the processed image and a string description.

    Args:
        image_path (str): Path to the image file.
        model_type (str): Type of model to use for detection ('yolov5' or 'detic').
        threshold (float): Detection threshold.
        show_confidence (bool): Whether to show confidence scores on the output image.

    Returns:
        tuple: A tuple containing the processed Image.Image and a string of detected objects.
    """

    detector = ObjectDetector()
    detector.load_model(model_type)
    image = detector.process_image(image_path)
    detected_objects_string, detected_objects_list = detector.detect_objects(image, threshold=threshold)
    image_with_boxes = detector.draw_boxes(image, detected_objects_list, show_confidence=show_confidence)
    return image_with_boxes, detected_objects_string


# Example usage
if __name__ == "__main__":

    # 'Sample_Images' is the folder conatining sample images for demo.
    image_path = get_path('horse.jpg', 'Sample_Images')
    processed_image, objects_string = detect_and_draw_objects(image_path,
                                                              model_type='detic',
                                                              threshold=0.2,
                                                              show_confidence=False)
    show_image(processed_image)
    print("Detected Objects:", objects_string)
